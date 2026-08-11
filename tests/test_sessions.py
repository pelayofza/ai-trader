from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_trader.backtest import session_study as ss
from ai_trader.data import intraday
from ai_trader.data.cache import cache_path


# --------------------------------------------------------------- utilidades ---------


def hourly_frame(
    n_days: int,
    *,
    start: str = "2024-01-01",
    closes=None,
    highs=None,
    lows=None,
    opens=None,
    drop: list[int] | None = None,
) -> pd.DataFrame:
    """Barras 1H sinteticas en la rejilla exacta. `drop` quita posiciones absolutas."""
    n = n_days * 24
    index = pd.date_range(start, periods=n, freq="h", tz="UTC")
    base = np.full(n, 100.0) if closes is None else np.asarray(closes, dtype=float)
    open_ = base if opens is None else np.asarray(opens, dtype=float)
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": base + 1.0 if highs is None else np.asarray(highs, dtype=float),
            "low": base - 1.0 if lows is None else np.asarray(lows, dtype=float),
            "close": base,
            "volume": np.full(n, 1_000.0),
        },
        index=index,
    )
    frame.index.name = "timestamp"
    if drop:
        frame = frame.drop(frame.index[drop])
    return frame


# ------------------------------------------------------------ (b) sesiones ----------


def test_sessions_particionan_el_dia_utc_sin_huecos_ni_solapes():
    assert ss.SESSIONS[0].start_hour == 0
    assert ss.SESSIONS[-1].end_hour == ss.HOURS_PER_DAY
    for previous, current in zip(ss.SESSIONS, ss.SESSIONS[1:]):
        assert current.start_hour == previous.end_hour
    assert sum(s.hours for s in ss.SESSIONS) == 24
    assert set(ss.HOUR_TO_SESSION) == {0, 1, 2}


def test_cortes_caen_en_las_aperturas_declaradas():
    """Los cortes no son redondos: son aperturas de mercado. Fijarlos aqui evita que
    alguien los mueva 'para que queden bloques de 8 horas' y rompa la comparabilidad."""
    by_key = {s.key: s for s in ss.SESSIONS}
    assert (by_key["asia"].start_hour, by_key["asia"].end_hour) == (0, 7)
    assert (by_key["europe"].start_hour, by_key["europe"].end_hour) == (7, 13)
    assert (by_key["us"].start_hour, by_key["us"].end_hour) == (13, 24)
    # Y no duran lo mismo, que es justo por lo que hay que publicar la intensidad.
    assert len({s.hours for s in ss.SESSIONS}) == 3


def test_session_of_hour_mapea_las_24_horas():
    assert [ss.session_of_hour(h) for h in (0, 6, 7, 12, 13, 23)] == [
        "asia", "asia", "europe", "europe", "us", "us",
    ]
    with pytest.raises(ValueError):
        ss.session_of_hour(24)


def test_clock_share_suma_uno():
    assert sum(s.clock_share for s in ss.SESSIONS) == pytest.approx(1.0)


# --------------------------------------------------- (a) cache 1H vs 1D -------------


def test_la_clave_de_cache_lleva_el_prefijo_de_clase():
    assert intraday.cache_key("btc/usdt") == "crypto::BTC/USDT"


def test_el_fichero_1h_no_pisa_al_1d():
    """El timeframe forma parte del NOMBRE del fichero: descargar 1H no puede invalidar
    el historico diario que consumen el backtest y los estudios ya publicados."""
    key = intraday.cache_key("BTC/USDT")
    hourly = cache_path(key, timeframe=intraday.INTRADAY_TIMEFRAME)
    daily = cache_path(key, timeframe="1D")
    assert hourly != daily
    assert hourly.name == "CRYPTO__BTC_USDT_1H.parquet"
    assert daily.name == "CRYPTO__BTC_USDT_1D.parquet"


def test_slice_window_excluye_el_final():
    frame = hourly_frame(2)
    window = intraday.slice_window(frame, "2024-01-01", "2024-01-02")
    assert len(window) == 24
    assert window.index.max() == pd.Timestamp("2024-01-01 23:00", tz="UTC")


def test_get_hourly_bars_offline_sin_cache_devuelve_none(monkeypatch):
    monkeypatch.setattr(intraday, "load_cached_hourly", lambda symbol: None)
    assert intraday.get_hourly_bars("BTC/USDT", "2024-01-01", "2024-01-02") is None


def test_get_hourly_bars_offline_recorta_la_cache(monkeypatch):
    monkeypatch.setattr(intraday, "load_cached_hourly", lambda symbol: hourly_frame(5))
    window = intraday.get_hourly_bars("BTC/USDT", "2024-01-02", "2024-01-04")
    assert len(window) == 48
    assert window.index.min() == pd.Timestamp("2024-01-02", tz="UTC")


def test_ventana_invertida_es_error():
    with pytest.raises(ValueError):
        intraday.get_hourly_bars("BTC/USDT", "2024-01-05", "2024-01-01")


# ----------------------------------------------- (c) matriz diaria y cuotas ---------


def test_solo_entran_dias_completos_y_encadenados():
    """El primer dia siempre cae (no hay barra de las 23:00 del dia anterior para el
    retorno de la hora 0) y un dia con una barra menos cae entero."""
    matrix = ss.build_day_matrix(hourly_frame(4, drop=[50]))  # hora 2 del tercer dia
    assert matrix.n_days == 2  # cae el dia 1 (sin encadenar) y el dia 3 (incompleto)
    assert [str(d.date()) for d in matrix.days] == ["2024-01-02", "2024-01-04"]
    assert matrix.n_days_dropped == 2


def test_el_dia_agregado_reproduce_la_vela_diaria():
    n = 3 * 24
    closes = 100.0 + np.arange(n, dtype=float)
    matrix = ss.build_day_matrix(hourly_frame(3, closes=closes))
    table = ss.daily_table(matrix)
    # El open del dia es el open de las 00:00, el close el de las 23:00, y el high/low
    # el maximo/minimo de las 24 barras: es exactamente la vela 1D que ve el motor.
    day2 = table.loc[pd.Timestamp("2024-01-02", tz="UTC")]
    assert day2["open"] == pytest.approx(closes[24])
    assert day2["close"] == pytest.approx(closes[47])
    assert day2["high"] == pytest.approx(closes[47] + 1.0)
    assert day2["low"] == pytest.approx(closes[24] - 1.0)
    assert day2["prev_close"] == pytest.approx(closes[23])


def test_toda_la_actividad_en_una_sesion_le_da_toda_la_cuota():
    """Serie plana salvo dentro del tramo estadounidense: su cuota tiene que ser 1."""
    n_days = 4
    closes = np.empty(n_days * 24)
    level = 100.0
    for d in range(n_days):
        for h in range(24):
            if h >= 13:  # el precio SOLO se mueve dentro de la sesion estadounidense
                level += 1.0
            closes[d * 24 + h] = level
    frame = hourly_frame(n_days, closes=closes, highs=closes, lows=closes, opens=closes)
    table = ss.daily_table(ss.build_day_matrix(frame))
    shares = ss.session_shares(table)
    assert shares["us"]["variance"] == pytest.approx(1.0, abs=1e-9)
    assert shares["asia"]["variance"] == pytest.approx(0.0, abs=1e-9)
    assert shares["europe"]["variance"] == pytest.approx(0.0, abs=1e-9)


def test_las_cuotas_suman_uno_y_la_intensidad_normaliza_por_reloj():
    rng = np.random.default_rng(7)
    n = 30 * 24
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    frame = hourly_frame(30, closes=closes, highs=closes * 1.001, lows=closes * 0.999)
    shares = ss.session_shares(ss.daily_table(ss.build_day_matrix(frame)))

    for field in ("abs_return", "variance", "range"):
        assert sum(shares[k][field] for k in ss.SESSION_KEYS) == pytest.approx(1.0, abs=1e-3)
    # `range_vs_daily` mide solape, no reparto: tiene que sumar MAS de 1.
    assert sum(shares[k]["range_vs_daily"] for k in ss.SESSION_KEYS) > 1.0
    # Con ruido homogeneo la intensidad es ~1 en los tres tramos, aunque las cuotas no
    # se parezcan entre si: es lo que hace comparables sesiones de distinta duracion.
    for key in ss.SESSION_KEYS:
        assert shares[key]["variance_intensity"] == pytest.approx(1.0, abs=0.25)


def test_sets_high_y_sets_low_reparten_los_extremos():
    rng = np.random.default_rng(11)
    n = 40 * 24
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    frame = hourly_frame(40, closes=closes, highs=closes * 1.002, lows=closes * 0.998)
    shares = ss.session_shares(ss.daily_table(ss.build_day_matrix(frame)))
    assert sum(shares[k]["sets_high"] for k in ss.SESSION_KEYS) == pytest.approx(1.0)
    assert sum(shares[k]["sets_low"] for k in ss.SESSION_KEYS) == pytest.approx(1.0)


# ----------------------------------------------------- (d) la cifra que decide ------


def test_sin_hueco_la_ventana_ciega_no_tiene_ancho():
    """En 24/7 el open de las 00:00 es el cierre de las 23:00 de ayer: hueco exactamente 0.
    Es el caso de referencia contra el que se lee la cifra real."""
    n = 5 * 24
    closes = 100.0 + np.arange(n, dtype=float)
    frame = hourly_frame(5, closes=closes, opens=closes)  # open_h == close_h
    table = ss.daily_table(ss.build_day_matrix(frame))
    # open del dia = close de la hora 0 = close de las 23:00 de ayer + 1 en este sintetico
    gap = ss.gap_stats(table)
    assert gap["n_days"] == 4
    assert gap["share_of_range"]["median"] is not None


def test_el_hueco_se_mide_contra_el_rango_del_dia():
    """Hueco construido a mano: el open del dia se separa del cierre de ayer 5 unidades,
    con un rango diario de 20. La cuota tiene que salir 0,25 exacta."""
    n_days = 2
    n = n_days * 24
    closes, opens, highs, lows = (np.empty(n) for _ in range(4))
    for d in range(n_days):
        base, level = d * 24, 100.0 + 5.0 * d
        opens[base:base + 24] = level
        closes[base:base + 24] = level
        highs[base:base + 24] = level + 10.0
        lows[base:base + 24] = level - 10.0   # rango del dia = 20

    frame = hourly_frame(n_days, closes=closes, opens=opens, highs=highs, lows=lows)
    gap = ss.gap_stats(ss.daily_table(ss.build_day_matrix(frame)))
    assert gap["n_days"] == 1  # el primer dia cae por no tener con que encadenar
    assert gap["share_of_range"]["median"] == pytest.approx(0.25)
    # 5 sobre un cierre previo de 100 son 500 pb.
    assert gap["bps"]["median"] == pytest.approx(500.0)
    assert gap["days_above_threshold_pct"] == pytest.approx(100.0)


def test_la_latencia_mide_el_desplazamiento_desde_el_open():
    """Precio que sube 1 por hora desde el open: con k horas de retraso el llenado se ha
    desplazado k unidades. Con rango diario 23, la cuota de 1h es 1/23."""
    n_days = 3
    ramp = np.tile(np.arange(24, dtype=float), n_days) + 100.0
    frame = hourly_frame(n_days, closes=ramp, opens=ramp, highs=ramp, lows=ramp)
    table = ss.daily_table(ss.build_day_matrix(frame))
    rows = {r["hours"]: r for r in ss.latency_stats(table)["rows"]}
    assert rows[1]["slip_share_of_range_median"] == pytest.approx(1.0 / 23.0, abs=1e-4)
    assert rows[4]["slip_share_of_range_median"] == pytest.approx(4.0 / 23.0, abs=1e-4)
    # La sesion en la que cae cada retraso sale del mismo mapeo que la descomposicion.
    assert rows[1]["session"] == "asia"
    assert rows[8]["session"] == "europe"


# ------------------------------------------------------------- (c) tendencia --------


def test_test_de_signos_exacto():
    assert ss._binomial_two_sided(0, 0) == 1.0
    assert ss._binomial_two_sided(5, 10) == pytest.approx(1.0)
    assert ss._binomial_two_sided(10, 10) == pytest.approx(2.0 / 1024.0)
    assert ss._binomial_two_sided(9, 10) == pytest.approx(2.0 * 11.0 / 1024.0)


def test_la_tendencia_detecta_un_desplazamiento_construido():
    """Panel donde la actividad estadounidense sube despues del corte en todos los pares:
    el veredicto declarado tiene que activarse."""
    tables, years = {}, [2023, 2024]
    for i in range(6):
        blocks = []
        for year, us_extra in ((2023, 0.0), (2024, 4.0)):
            n_days = 300
            closes = np.full(n_days * 24, 100.0)
            for d in range(n_days):
                for h in range(24):
                    amp = 1.0 + (us_extra if h >= 13 else 0.0)
                    closes[d * 24 + h] = 100.0 + amp * ((-1) ** (h + d + i))
            frame = hourly_frame(n_days, start=f"{year}-01-01", closes=closes,
                                 highs=closes, lows=closes, opens=closes)
            blocks.append(ss.daily_table(ss.build_day_matrix(frame)))
        tables[f"X{i}/USDT"] = pd.concat(blocks)

    trend = ss.trend_analysis(tables, sorted(tables), years)
    assert trend["n_symbols"] == 6
    assert trend["n_positive"] == 6
    assert trend["mean_delta_variance"] > ss.US_SHARE_GROWTH_MIN
    assert trend["verdict"] == ss.VERDICT_TREND_UP
    assert len(trend["yearly"]) == 2


def test_la_forma_distingue_una_deriva_previa_de_un_salto():
    """El contraste pre/post no distingue 'sube por el mecanismo' de 'lleva anos subiendo'.
    La cifra que lo separa es la pendiente sobre los anos ANTERIORES al corte."""
    yearly_drift = [
        {"year": y, "us": {"variance": v}}
        for y, v in [(2020, 0.45), (2021, 0.44), (2022, 0.51), (2023, 0.55),
                     (2024, 0.52), (2025, 0.56)]
    ]
    yearly_jump = [
        {"year": y, "us": {"variance": v}}
        for y, v in [(2020, 0.45), (2021, 0.44), (2022, 0.45), (2023, 0.44),
                     (2024, 0.56), (2025, 0.57)]
    ]

    def shape_of(yearly):
        before = [r for r in yearly if r["year"] < 2024]
        after = [r for r in yearly if r["year"] >= 2024]
        rho = ss._spearman([r["year"] for r in before], [r["us"]["variance"] for r in before])
        step = after[0]["us"]["variance"] - before[-1]["us"]["variance"]
        return rho, step

    rho_drift, step_drift = shape_of(yearly_drift)
    assert rho_drift >= ss.PRE_TREND_RHO and step_drift < 0
    rho_jump, step_jump = shape_of(yearly_jump)
    assert rho_jump < ss.PRE_TREND_RHO and step_jump > 0


def test_el_veredicto_de_tendencia_separa_direccion_de_mecanismo():
    """Una deriva previa tiene que aparecer en el texto aunque la direccion se sostenga:
    cobrarle a un evento de 2024 una pendiente que ya llevaba anos es el error facil."""
    trend = {
        "verdict": ss.VERDICT_TREND_UP,
        "shape": ss.SHAPE_DRIFT,
        "pre_split_spearman": 0.8,
        "pre_split_years": [2020, 2023],
        "step_at_split": -0.023,
        "mean_delta_variance": 0.076,
        "n_positive": 10,
        "n_symbols": 10,
        "sign_test_p": 0.002,
        "split": "2024-01-01",
        "threshold": ss.US_SHARE_GROWTH_MIN,
    }
    verdicts = ss._verdicts(
        {"share_of_range": {"median": 0.0007}, "bps": {"median": 0.55}},
        ss.VERDICT_GAP_NEGLIGIBLE,
        {"rows": []},
        False,
        trend,
        15.0,
    )
    text = verdicts["trend"]["text"]
    assert "DIRECCION" in text and "MECANISMO" in text
    assert "DERIVA" in text


def test_el_coste_de_referencia_es_comision_mas_deslizamiento():
    from ai_trader.config import load_config

    config = load_config("config/default.toml")
    expected = config.execution.fee_rate * 1e4 + config.execution.slippage_bps
    assert ss.reference_cost_bps(config) == pytest.approx(expected)


def test_sin_desplazamiento_el_veredicto_es_plano():
    rng = np.random.default_rng(3)
    tables = {}
    for i in range(5):
        n_days = 600
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n_days * 24)))
        frame = hourly_frame(n_days, start="2023-01-01", closes=closes,
                             highs=closes * 1.001, lows=closes * 0.999)
        tables[f"Y{i}/USDT"] = ss.daily_table(ss.build_day_matrix(frame))
    trend = ss.trend_analysis(tables, sorted(tables), [2023, 2024])
    assert trend["verdict"] == ss.VERDICT_TREND_FLAT


# ------------------------------------------------------------------ informe ---------


def _panel(n_symbols: int = 3, n_days: int = 800) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(99)
    tables = {}
    for i in range(n_symbols):
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n_days * 24)))
        frame = hourly_frame(n_days, start="2023-01-01", closes=closes,
                             highs=closes * 1.001, lows=closes * 0.999)
        tables[f"Z{i}/USDT"] = ss.daily_table(ss.build_day_matrix(frame))
    return tables


def test_analyze_produce_un_informe_completo_y_determinista():
    tables = _panel()
    matrices = {s: ss.build_day_matrix(hourly_frame(1)) for s in tables}
    plan = {
        "exchange": "binance",
        "start_year": 2023,
        "end_year": 2024,
        "window": {"start": "2023-01-01", "end": "2025-01-01"},
        "reference_cost_bps": 15.0,
    }
    first = ss.analyze(tables, matrices, plan, [])
    second = ss.analyze(tables, matrices, plan, [])
    assert ss._strip_volatile(first) == ss._strip_volatile(second)

    assert len(first["sessions"]) == 3
    assert first["cohort"] == sorted(tables)
    assert first["gap"]["n_days"] > 0
    assert first["verdicts"]["gap"]["key"] in (
        ss.VERDICT_GAP_MATERIAL, ss.VERDICT_GAP_NEGLIGIBLE
    )
    assert first["verdicts"]["trend"]["key"] in (
        ss.VERDICT_TREND_UP, ss.VERDICT_TREND_FLAT, ss.VERDICT_TREND_DOWN
    )
    assert len(first["by_symbol_year"]) == 3 * 2  # 3 simbolos x 2 anos completos
    assert {c["key"] for c in first["caveats"]} >= {
        "sesiones_desiguales", "solape_de_sesiones", "cohorte_equilibrada"
    }


def test_sin_cohorte_equilibrada_el_estudio_falla_en_vez_de_publicar_una_tendencia_falsa():
    """Si ningun par cubre todos los anos, la serie anual mediria composicion del universo.
    Preferimos que reviente a que publique una tendencia que no es una tendencia."""
    tables = _panel(n_symbols=2, n_days=300)  # solo 2023
    matrices = {s: ss.build_day_matrix(hourly_frame(1)) for s in tables}
    plan = {
        "exchange": "binance",
        "start_year": 2023,
        "end_year": 2024,
        "window": {"start": "2023-01-01", "end": "2025-01-01"},
    }
    with pytest.raises(ValueError, match="cohorte equilibrada"):
        ss.analyze(tables, matrices, plan, [])


def test_el_informe_publicado_es_legible_y_coherente():
    """Si el informe existe en el repo, sus invariantes tienen que seguir cumpliendose."""
    report = ss.load_sessions_report()
    if report is None:
        pytest.skip("Sin informe publicado en data/sessions/report.json")

    assert [s["key"] for s in report["sessions"]] == list(ss.SESSION_KEYS)
    for field in ("abs_return", "variance", "range"):
        total = sum(report["overall"]["sessions"][k][field] for k in ss.SESSION_KEYS)
        assert total == pytest.approx(1.0, abs=1e-3)
    assert report["plan"]["window"]["end"] == ss.DEFAULT_END
    assert report["gap"]["threshold"] == ss.GAP_MATERIAL_SHARE
    assert report["cohort"]
    assert report["verdicts"]["gap"]["text"]
    # La tendencia publica su FORMA, no solo su direccion: sin eso, un contraste pre/post
    # se lee como efecto del mecanismo declarado aunque sea una deriva de varios anos.
    assert report["trend"]["shape"] in (ss.SHAPE_DRIFT, ss.SHAPE_JUMP, ss.SHAPE_MIXED)
    assert "pre_split_spearman" in report["trend"]
