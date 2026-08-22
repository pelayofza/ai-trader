"""
Tests de la puntuacion experta del reporte diario y de la estrategia que la usa.

Lo que hay que blindar aqui NO es "que la formula de este numero": los pesos son un juicio
afirmado y cambiarlos es legitimo. Lo que no puede cambiar en silencio son las cinco
propiedades que hacen que ese juicio no se contamine:

- **P30 no se puede usar.** Es el sesgo global que escribe el mismo redactor que respondio
  P01-P29. El lector no la carga y la tabla no la nombra, asi que no hay camino desde el
  benchmark hasta una orden. Es la unica de las 37 con esta propiedad y se comprueba por los
  dos lados.
- **El score no premia la cobertura.** Es una media ponderada, asi que dos activos con la
  misma respuesta y distinto numero de respuestas puntuan igual. Sin esto, el ranking del dia
  ordenaria por CUANTO se pudo medir de cada activo, que es el sesgo que el propio resumen
  del dia vigila con `diagnostico_cobertura_vs_media`.
- **La eleccion es de la estrategia, no del orden del universo.** Con las 24 lecturas del
  mismo signo, el umbral absoluto solo no elige: elige el fichero de config. El corte
  transversal es lo que lo arregla y hay un test con el dia real.
- **La horquilla depende de la senal.** Si no, "stop y take profit variables" seria una
  constante con nombre largo.
- **El stop nunca obliga al motor de riesgo a reescribirlo.** El techo de la config esta por
  debajo de `max_stop_distance_pct`, asi que el riesgo aprueba sin apretar y lo que dice el
  log de la estrategia es lo que se opera.

Los datos de la captura real (`data/signals_raw/`) estan fuera de git, asi que los tests que
los usan se saltan solos en un clon recien hecho. Los que fijan el comportamiento van con
respuestas construidas a mano y corren siempre.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ai_trader.observation.daily_report_scores import (
    BENCHMARK_QUESTION,
    CROWDING_QUESTIONS,
    DIRECTIONAL,
    MIN_COVERAGE,
    VOL_FALLBACK_ANNUAL_PCT,
    AssetScore,
    DailyReportProvider,
    daily_sigma_pct,
    score_asset,
    score_day,
    ticker_for,
)
from ai_trader.risk.engine import PortfolioState, RiskEngine, RiskLimits
from ai_trader.shared.schemas import Side
from ai_trader.signals.ai_reports import QUESTIONNAIRE, QUESTIONNAIRE_ID, load_day_answers
from ai_trader.strategies.daily_report_expert import (
    DailyReportExpertConfig,
    DailyReportExpertStrategy,
)
from tests.conftest import build_bars

REPO_ROOT = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)
CUTOFF = "2026-08-22T06:00:00Z"


class FrozenClock:
    def __init__(self, moment: datetime = NOW) -> None:
        self._moment = moment

    def now(self) -> datetime:
        return self._moment


# --- utillería de respuestas a mano ---------------------------------------------------


def answer(option: str, value: float | None, raw: float | None = None) -> dict:
    state = "medido" if value is not None else "sin_datos"
    return {
        "id_opcion": option,
        "valor": value,
        "valor_crudo": raw,
        "estado": state,
        "disponible": 1 if value is not None else 0,
    }


def answers_for(values: dict[str, float], **extra: dict) -> dict:
    """Un fichero de respuestas minimo: los ids que se pasan y nada mas."""
    built = {qid: answer("opt", value) for qid, value in values.items()}
    built.update(extra)
    return built


def day_with(assets: dict[str, dict], *, cutoff: str = CUTOFF) -> dict:
    return {
        "date": "2026-08-22",
        "questionnaire": QUESTIONNAIRE_ID,
        "cutoff_utc": cutoff,
        "assets": {
            ticker: {
                "answers": answers,
                "cutoff_utc": cutoff,
                "anchor_usd": 100.0,
                "anchor_ts": cutoff,
                "questionnaire": QUESTIONNAIRE_ID,
            }
            for ticker, answers in assets.items()
        },
    }


def full_marks(value: float, *, exclude: tuple[str, ...] = ()) -> dict:
    """Las 32 preguntas de direccion respondidas con el mismo valor."""
    return answers_for({q: value for q in DIRECTIONAL if q not in exclude})


# --- P30: el benchmark que no puede llegar a una orden --------------------------------


class TestElBenchmarkNoDecide:
    def test_la_tabla_de_pesos_no_nombra_p30(self):
        assert BENCHMARK_QUESTION not in DIRECTIONAL

    def test_el_cuestionario_confirma_que_p30_es_el_benchmark(self):
        """Que P30 sea la excluida no es una convencion de este modulo: lo dice el propio
        contrato, y si algun dia dejara de decirlo, esto tiene que romperse aqui."""
        questionnaire = json.loads((REPO_ROOT / QUESTIONNAIRE).read_text(encoding="utf-8"))
        p30 = next(q for q in questionnaire["preguntas"] if q["id"] == BENCHMARK_QUESTION)
        assert p30["sumable"] is False

    def test_aunque_p30_llegue_en_las_respuestas_no_mueve_el_score(self):
        """La defensa de fondo: el lector no la carga, pero si alguien construyera el dict a
        mano con P30 dentro, la tabla sigue sin mirarla."""
        base = score_asset("X", full_marks(1.0))
        with_benchmark = score_asset("X", {**full_marks(1.0), "P30": answer("muy_alcista", 2)})
        assert base is not None and with_benchmark is not None
        assert base.score == pytest.approx(with_benchmark.score)

    def test_el_lector_no_saca_p30_del_fichero_del_dia(self):
        day = load_day_answers(REPO_ROOT)
        if day is None:
            pytest.skip("sin captura del reporte diario en este clon")
        for payload in day["assets"].values():
            assert BENCHMARK_QUESTION not in payload["answers"]


# --- el score, y el sesgo que no puede tener ------------------------------------------


class TestElScoreEsUnaMediaPonderada:
    def test_media_y_no_suma_el_numero_de_respuestas_no_premia(self):
        """Dos activos que contestan LO MISMO puntuan igual aunque uno conteste 29 preguntas
        y el otro 12. Es la propiedad que impide que el ranking del dia ordene por cuanto se
        pudo medir de cada uno."""
        todas = score_asset("TODAS", full_marks(2.0, exclude=CROWDING_QUESTIONS))
        pocas = score_asset("POCAS", answers_for(dict.fromkeys(
            ["P01", "P02", "P03", "P04", "P13", "P19", "P20", "P22", "P23", "P24", "P28", "P36"],
            2.0,
        )))
        assert todas is not None and pocas is not None
        assert todas.score == pytest.approx(pocas.score)
        assert todas.coverage > pocas.coverage

    def test_el_score_vive_en_menos_uno_mas_uno(self):
        """Los extremos se toman SIN las tres contrarian: con ellas dentro, contestar +2 a
        todo se contradice a si mismo —el dia perfecto no puede tener a la vez el mejor flujo
        y el peor funding— y el maximo se queda en 0,83. Que no llegue a 1 no es un fallo de
        escala, es la polaridad invertida haciendo su trabajo."""
        assert score_asset("X", full_marks(2.0, exclude=CROWDING_QUESTIONS)).score == (
            pytest.approx(1.0)
        )
        assert score_asset("X", full_marks(-2.0, exclude=CROWDING_QUESTIONS)).score == (
            pytest.approx(-1.0)
        )
        assert score_asset("X", full_marks(2.0)).score < 1.0

    def test_por_debajo_del_piso_de_cobertura_no_se_puntua(self):
        """No puntua mal: no puntua. Devolver un score de dos respuestas y dejar que la
        estrategia decida seria justo lo que el piso existe para impedir."""
        assert score_asset("X", answers_for({"P01": 2.0, "P27": 2.0})) is None

    def test_el_piso_es_de_cobertura_PONDERADA_no_de_cuenta(self):
        """Doce preguntas de peso alto pasan; doce de peso bajo no. Contar preguntas trataria
        igual a un flujo de ETF y a la estacionalidad del mes."""
        pesadas = ["P02", "P13", "P20", "P23", "P03", "P10", "P12", "P28", "P36"]
        assert score_asset("X", answers_for(dict.fromkeys(pesadas, 1.0))) is not None
        ligeras = ["P01", "P06", "P26", "P27", "P37", "P14", "P17"]
        assert score_asset("X", answers_for(dict.fromkeys(ligeras, 1.0))) is None


class TestLasTresContrarian:
    @pytest.mark.parametrize("qid", CROWDING_QUESTIONS)
    def test_la_aglomeracion_RESTA(self, qid):
        """RSI arriba, interes abierto extremo y funding positivo son razones para desconfiar
        del largo, no para reforzarlo. Es la unica interpretacion que este modulo anade sobre
        la escala del cuestionario, y por eso se comprueba una por una."""
        neutral = score_asset("X", {**full_marks(1.0), qid: answer("neutro", 0.0)})
        crowded = score_asset("X", {**full_marks(1.0), qid: answer("extremo", 2.0)})
        assert crowded.score < neutral.score

    def test_la_cobertura_de_la_aglomeracion_se_publica(self):
        """Con una sola de las tres medida, la lectura puede valer +1,00 sin que se sepa nada
        de las otras dos. Quien la use para acortar un objetivo necesita poder escalarla."""
        solo_rsi = score_asset("X", {
            **full_marks(1.0, exclude=("P16", "P31")),
            "P25": answer("muy_alto", 2.0),
        })
        assert solo_rsi.crowding == pytest.approx(1.0)
        assert 0.0 < solo_rsi.crowding_coverage < 1.0


class TestLosModuladores:
    def test_la_beta_escala_el_bloque_de_mercado_y_solo_ese(self):
        alta = score_asset("X", {**full_marks(1.0), "P35": answer("muy_alta", 2.0)})
        baja = score_asset("X", {**full_marks(1.0), "P35": answer("muy_baja", -2.0)})
        assert alta.beta_scale > baja.beta_scale
        assert alta.blocks["mercado"] > baja.blocks["mercado"]
        assert alta.blocks["tecnico"] == pytest.approx(baja.blocks["tecnico"], abs=0.02)

    def test_p35_no_vota_en_la_direccion(self):
        """La beta dice CUANTO hereda el activo del mercado, no si sube. Si votase, un activo
        volatil puntuaria alto por ser volatil."""
        assert "P35" not in DIRECTIONAL

    def test_la_profundidad_solo_puede_restar(self):
        fina = score_asset("X", {**full_marks(1.0), "P34": answer("muy_baja", -2.0)})
        profunda = score_asset("X", {**full_marks(1.0), "P34": answer("muy_alta", 2.0)})
        sin_dato = score_asset("X", full_marks(1.0))
        assert fina.depth_factor < 1.0
        assert profunda.depth_factor == 1.0 == sin_dato.depth_factor


class TestLaSigmaDeLaHorquilla:
    def test_prefiere_el_numero_crudo_a_la_categoria(self):
        """`baja` es un tramo entero y 55,0 es un numero. Volver a bucketizar teniendo el
        crudo al lado seria reintroducir el error que la v2 del cuestionario vino a quitar."""
        sigma, source = daily_sigma_pct({"P32": answer("baja", -1.0, raw=55.0)})
        assert sigma == pytest.approx(55.0 / (365 ** 0.5), rel=1e-6)
        assert source == "realizada"

    def test_sin_crudo_cae_a_la_categoria_y_lo_dice(self):
        sigma, source = daily_sigma_pct({"P32": answer("muy_alta", 2.0)})
        assert sigma == pytest.approx(VOL_FALLBACK_ANNUAL_PCT["muy_alta"] / (365 ** 0.5))
        assert source == "realizada_categoria"

    def test_sin_ninguna_de_las_dos_usa_el_respaldo_y_lo_dice(self):
        _, source = daily_sigma_pct({})
        assert source == "respaldo"

    def test_realizada_e_implicita_se_promedian(self):
        sigma, source = daily_sigma_pct({
            "P32": answer("baja", -1.0, raw=40.0),
            "P33": answer("alta", 1.0, raw=80.0),
        })
        assert sigma == pytest.approx(60.0 / (365 ** 0.5))
        assert source == "realizada+implicita"


# --- la estrategia --------------------------------------------------------------------


def make_strategy(day: dict, *, now: datetime = NOW, **params) -> DailyReportExpertStrategy:
    strategy = DailyReportExpertStrategy(DailyReportExpertConfig(**params))
    strategy.attach_daily_report_provider(DailyReportProvider(day, FrozenClock(now)))
    return strategy


BARS = build_bars([100.0] * 5)


class TestLaEstrategiaDegradaANADA:
    def test_sin_proveedor_no_emite_y_no_revienta(self):
        assert DailyReportExpertStrategy().generate_signal("BTC/USDT", BARS) is None

    def test_sin_captura_no_emite(self):
        assert make_strategy(None).generate_signal("BTC/USDT", BARS) is None

    def test_un_reporte_caducado_no_opera(self):
        day = day_with({"BTC": full_marks(2.0)})
        fresco = make_strategy(day)
        caducado = make_strategy(day, now=NOW + timedelta(hours=40))
        assert fresco.generate_signal("BTC/USDT", BARS) is not None
        assert caducado.generate_signal("BTC/USDT", BARS) is None

    def test_sin_hora_de_corte_se_trata_como_caducado(self):
        """Un fichero sin corte legible no es un fichero fresco: es uno del que no se sabe la
        edad, y para operar con el hay que saberla."""
        day = day_with({"BTC": full_marks(2.0)}, cutoff="")
        assert make_strategy(day).generate_signal("BTC/USDT", BARS) is None

    def test_otro_cuestionario_no_se_puntua(self):
        """Una v3 con otros ids no se puntua mal: no se puntua. P17 podria significar otra
        cosa y el score saldria con la misma pinta."""
        day = {**day_with({"BTC": full_marks(2.0)}), "questionnaire": "cuestionario_cripto_v9"}
        assert score_day(day) == {}
        assert make_strategy(day).generate_signal("BTC/USDT", BARS) is None

    def test_sin_barras_no_hay_precio_de_entrada(self):
        day = day_with({"BTC": full_marks(2.0)})
        assert make_strategy(day).generate_signal("BTC/USDT", None) is None

    @pytest.mark.parametrize("symbol", ["PM::algo", "AAPL", ""])
    def test_lo_que_no_es_un_par_de_cripto_no_se_soporta(self, symbol):
        assert ticker_for(symbol) is None
        assert DailyReportExpertStrategy().supports_symbol(symbol) is False


class TestLadoYEleccion:
    def test_hace_falta_conviccion_absoluta(self):
        """Ser el mejor de un dia plano no basta: si el dia no dice nada, el primero de la
        lista tampoco."""
        day = day_with({"BTC": full_marks(0.1)})
        assert make_strategy(day, min_abs_score=0.5).generate_signal("BTC/USDT", BARS) is None

    def test_hace_falta_ESTAR_ENTRE_LOS_MEJORES(self):
        """El espejo del anterior, y el que de verdad importa: con todo el dia del mismo
        signo, el umbral absoluto lo pasan los 24 y quien elige acaba siendo el orden del
        fichero de config."""
        day = day_with({t: full_marks(v) for t, v in
                        (("AAA", 2.0), ("BBB", 1.6), ("CCC", 1.2), ("DDD", 0.8))})
        strategy = make_strategy(day, top_n=2, min_abs_score=0.1)
        emitidas = [
            t for t in ("AAA", "BBB", "CCC", "DDD")
            if strategy.generate_signal(f"{t}/USDT", BARS) is not None
        ]
        assert emitidas == ["AAA", "BBB"]

    def test_el_corto_sale_con_la_horquilla_del_reves(self):
        day = day_with({"AAA": full_marks(-2.0), "BBB": full_marks(1.0)})
        signal = make_strategy(day, min_abs_score=0.1).generate_signal("AAA/USDT", BARS)
        assert signal is not None and signal.side is Side.SELL
        assert signal.stop_loss > signal.entry_price > signal.take_profit

    def test_allow_short_false_deja_pasar_el_corto(self):
        day = day_with({"AAA": full_marks(-2.0)})
        strategy = make_strategy(day, min_abs_score=0.1, allow_short=False)
        assert strategy.generate_signal("AAA/USDT", BARS) is None


class TestLaHorquillaDependeDeLaSenal:
    def _signal(self, answers, **params):
        day = day_with({"AAA": answers})
        return make_strategy(day, min_abs_score=0.1, **params).generate_signal("AAA/USDT", BARS)

    def test_mas_volatilidad_es_mas_stop(self):
        """Es la mitad de "stop y take profit variables": la anchura sale del propio reporte
        (P32/P33) y no de una constante."""
        tranquilo = self._signal({**full_marks(1.0), "P32": answer("baja", -1.0, raw=30.0)})
        agitado = self._signal({**full_marks(1.0), "P32": answer("muy_alta", 2.0, raw=120.0)})
        assert agitado.features["stop_pct"] > tranquilo.features["stop_pct"]

    def test_mas_conviccion_es_mas_objetivo(self):
        """Y la otra mitad: el multiplo al que se pone el objetivo sube con la lectura."""
        flojo = self._signal(full_marks(0.5))
        fuerte = self._signal(full_marks(2.0))
        assert fuerte.features["reward_multiple"] > flojo.features["reward_multiple"]

    def test_el_riesgo_de_evento_acorta_el_objetivo(self):
        tranquilo = self._signal({
            **full_marks(1.0), "P28": answer("ninguno", 2.0), "P29": answer("ninguno", 1.0),
        })
        cargado = self._signal({
            **full_marks(1.0), "P28": answer("muchos", -2.0), "P29": answer("varios", -2.0),
        })
        assert cargado.features["event_risk"] > tranquilo.features["event_risk"]
        assert cargado.features["reward_multiple"] < tranquilo.features["reward_multiple"]

    def test_la_aglomeracion_en_contra_acorta_el_objetivo_del_largo(self):
        limpio = self._signal({
            **full_marks(1.0, exclude=CROWDING_QUESTIONS),
            **{q: answer("neutro", 0.0) for q in CROWDING_QUESTIONS},
        })
        lleno = self._signal({
            **full_marks(1.0, exclude=CROWDING_QUESTIONS),
            **{q: answer("extremo", 2.0) for q in CROWDING_QUESTIONS},
        })
        assert lleno.features["reward_multiple"] < limpio.features["reward_multiple"]

    def test_el_suelo_y_el_techo_del_stop_se_respetan(self):
        quieto = self._signal({**full_marks(1.0), "P32": answer("muy_baja", -2.0, raw=1.0)})
        loco = self._signal({**full_marks(1.0), "P32": answer("muy_alta", 2.0, raw=900.0)})
        assert quieto.features["stop_pct"] == pytest.approx(DailyReportExpertConfig().min_stop_pct)
        assert loco.features["stop_pct"] == pytest.approx(DailyReportExpertConfig().max_stop_pct)

    def test_un_techo_por_encima_del_limite_del_riesgo_no_se_construye(self):
        """Un stop mas ancho que `max_stop_distance_pct` no seria mas permisivo: el motor lo
        apretaria y el log de la estrategia diria una horquilla que no es la que se opera."""
        with pytest.raises(ValueError):
            DailyReportExpertConfig(max_stop_pct=20.0)


class TestElRiesgoNoTieneQueReescribirNada:
    @pytest.mark.parametrize("vol", [1.0, 30.0, 80.0, 200.0, 900.0])
    def test_ningun_stop_dispara_el_aviso_de_apriete(self, vol):
        """Propiedad, no caso: para cualquier volatilidad del reporte, el motor aprueba sin
        avisos. Si algun dia deja de ser cierto, lo que se opera y lo que se registra dejan de
        coincidir."""
        day = day_with({"AAA": {**full_marks(2.0), "P32": answer("x", 0.0, raw=vol)}})
        signal = make_strategy(day, min_abs_score=0.1).generate_signal("AAA/USDT", BARS)
        decision = RiskEngine(RiskLimits()).evaluate(
            signal=signal, portfolio_state=PortfolioState(open_positions=[]),
        )
        assert decision.approved and not decision.warnings
        assert decision.stop_loss == pytest.approx(signal.stop_loss)
        assert decision.take_profit == pytest.approx(signal.take_profit)

    def test_la_confianza_minima_de_la_estrategia_supera_la_del_riesgo(self):
        """El umbral que decide tiene que ser el de la estrategia. Si la confianza minima que
        puede emitir cayera por debajo de `min_confidence_per_trade`, quien estaria filtrando
        seria el motor de riesgo y sus rechazos llenarian el diario."""
        day = day_with({"AAA": full_marks(0.4)})
        signal = make_strategy(day, min_abs_score=0.1).generate_signal("AAA/USDT", BARS)
        assert signal.confidence >= RiskLimits().min_confidence_per_trade


# --- contra la captura real, que puede no estar -----------------------------------------


def real_day() -> dict:
    day = load_day_answers(REPO_ROOT)
    if day is None:
        pytest.skip("sin captura del reporte diario en este clon")
    return day


class TestContraLaCapturaReal:
    def test_el_dia_entero_se_puntua(self):
        day = real_day()
        scores = score_day(day)
        assert scores, "ningun activo puntuable en la ultima captura"
        assert len(scores) == len(day["assets"])
        assert all(isinstance(s, AssetScore) for s in scores.values())
        assert all(s.coverage >= MIN_COVERAGE for s in scores.values())
        assert all(-1.0 <= s.score <= 1.0 for s in scores.values())

    def test_los_puestos_son_una_permutacion_completa(self):
        scores = score_day(real_day())
        n = len(scores)
        assert sorted(s.rank_long for s in scores.values()) == list(range(1, n + 1))
        assert sorted(s.rank_short for s in scores.values()) == list(range(1, n + 1))

    def test_la_estrategia_propone_como_mucho_top_n(self):
        """El test que justifica el corte transversal. La captura del 2026-08-22 sale con las
        24 lecturas positivas: sin ordenacion, esto seria 24 y el runner se quedaria con los
        cinco primeros del universo."""
        day = real_day()
        cutoff = datetime.fromisoformat(day["cutoff_utc"].replace("Z", "+00:00"))
        strategy = make_strategy(day, now=cutoff + timedelta(hours=6))
        emitidas = [
            t for t in day["assets"]
            if strategy.generate_signal(f"{t}/USDT", BARS) is not None
        ]
        assert 0 < len(emitidas) <= DailyReportExpertConfig().top_n

    def test_la_config_por_defecto_arranca_con_esta_estrategia_y_solo_con_ella(self):
        """La prioridad forzada es una decision escrita en `config/default.toml`, no un efecto
        lateral del orden del registro. Si alguien reactiva otra familia, este test lo dice."""
        from ai_trader.config import load_config

        specs = load_config(REPO_ROOT / "config" / "default.toml").strategies
        assert [s.type for s in specs] == ["daily_report_expert"]
