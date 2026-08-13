"""
Tests del radar tematico: quince numeros mas, y ni un digito distinto en los seis de antes.

Lo que hay que blindar aqui, por orden de importancia:

- **Los seis numeros de siempre no cambian NI UN DIGITO.** Se compara `float.hex()`, no
  `approx`: la subclase existe precisamente para que esa invariancia sea estructural, y un
  test que la compruebe con tolerancia no comprobaria nada.
- **La aritmetica de cobertura por tema**, que es donde vive la unica decision de diseno
  nueva: el suelo del denominador. Una sola fuente no abre ningun tema.
- **Ninguna puerta tematica bloquea por falta de datos.** El mismo invariante de la global,
  y por el mismo cuerpo.
- **La tabla clasifica las TREINTA.** Anadir una fuente al catalogo y no darle tema falla.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.observation.signal_radar import (
    ASSET_SIGNAL_FEATURES,
    MAX_STALE_DAYS,
    MIN_SIGNAL_COVERAGE,
    SIGNAL_FEATURES,
    SignalRadarProvider,
)
from ai_trader.observation.signal_themes import (
    MAX_THEMES_PER_SOURCE,
    MIN_THEME_SOURCES,
    THEME_FEATURES,
    THEME_NAMES,
    THEMELESS,
    THEMES,
    ThemedSignalRadarProvider,
    ThemeSpec,
    _check_themes,
    effective_denominator,
    theme_features,
    theme_reading,
    themed_gate_reason,
)
from ai_trader.shared.clock import HistoricalClock
from ai_trader.shared.signals import DAY, ENTITY, OBSERVED
from ai_trader.signals.catalog import CATALOG
from ai_trader.signals.normalize import Z_CLIP

UTC = timezone.utc

ENTITIES = ("BTC", "ETH", "SOL", "ADA", "XRP", "DOGE")
SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT")


def at(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(tzinfo=UTC)


def frame(entity: str, days, **columns) -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [[entity] * len(days), pd.DatetimeIndex(days, tz="UTC")], names=[ENTITY, DAY]
    )
    out = pd.DataFrame(columns, index=index)
    out[OBSERVED] = 1
    return out


def _series(entities=ENTITIES, n=40, start="2026-01-01", **columns) -> pd.DataFrame:
    """Una fuente continua con historia suficiente para que la z propia exista."""
    days = pd.date_range(start, periods=n, tz="UTC")
    blocks = []
    for i, entity in enumerate(entities):
        scaled = {
            name: list(np.linspace(lo, hi, n) * (i + 1)) for name, (lo, hi) in columns.items()
        }
        blocks.append(frame(entity, days, **scaled))
    return pd.concat(blocks).sort_index()


def all_frames() -> dict[str, pd.DataFrame]:
    """
    Un frame por cada una de las tres codificaciones y por cada uno de los cinco temas.

    No es un decorado: la invariancia de los seis numeros solo significa algo si el radar
    esta recorriendo de verdad los tres caminos de `_readings_for_source` (serie continua,
    calendario de eventos y mapa de precios) y los dos bloques (activo y mercado).
    """
    days = pd.date_range("2026-01-01", periods=40, tz="UTC")
    return {
        # continuas de activo
        "github_activity": _series(commits=(1.0, 10.0), contributors=(1.0, 4.0)),
        "deribit_volatility": _series(
            entities=("BTC", "ETH"),
            dvol_index=(40.0, 70.0),
            skew_25d=(-5.0, 5.0),
            atm_iv_30d=(35.0, 65.0),
            iv_term_slope=(-2.0, 3.0),
        ),
        "wikipedia_pageviews": _series(
            pageviews=(1000.0, 9000.0), pageviews_lang_concentration=(0.2, 0.8)
        ),
        "etf_flows": _series(
            entities=("BTC",),
            etf_netflow_usd=(-2e8, 5e8),
            etf_issuer_dispersion=(0.1, 0.9),
            etf_issuers_reporting=(3.0, 9.0),
        ),
        # continua de mercado
        "fred_macro": _series(
            entities=("DXY", "VIX", "DGS10"), macro_value=(90.0, 110.0), macro_change_1d=(-1.0, 1.0)
        ),
        # eventos de mercado
        "defillama_hacks": frame("*", ["2026-01-20"], hack_amount_usd=[3e8], hack_count=[1.0]),
        "macro_calendar": frame(
            "*", ["2026-02-12", "2026-03-18"], fomc_meeting=[1.0, 1.0], cpi_release=[0.0, 0.0]
        ),
        "federal_register": frame(
            "*", ["2026-01-18", "2026-02-02"], fedreg_documents=[4.0, 7.0], fedreg_rules=[1.0, 2.0]
        ),
        # evento de activo
        "cex_listings": pd.concat(
            [
                frame("SOL", ["2026-02-03"], listing_change=[1.0], listing_warning=[0.0]),
                frame("ADA", ["2026-01-28"], listing_change=[-1.0], listing_warning=[1.0]),
            ]
        ).sort_index(),
        # mapa de precios de activo
        "hyperliquid_liqmap": pd.concat(
            [
                frame(
                    e,
                    days[-3:],
                    liq_cluster_distance_pct=[-3.0, -2.5, -2.0],
                    liq_cluster_notional_usd=[2e8, 2.2e8, 2.4e8],
                )
                for e in ("BTC", "ETH", "SOL")
            ]
        ).sort_index(),
    }


# ------------------------------------------------------- la invariancia de los seis ----


class TestInvariancia:
    def test_las_seis_features_de_siempre_no_cambian_ni_un_digito(self):
        """
        El test que sostiene toda la evolucion. `float.hex()` y no `pytest.approx`: la
        exigencia es byte a byte, porque si esas seis se mueven deja de reproducirse
        `data/transfer/units_ai_v3.json` y con el la celda de control del barrido de rho.
        """
        frames = all_frames()
        base_clock = HistoricalClock(at("2026-02-05"))
        themed_clock = HistoricalClock(at("2026-02-05"))
        base = SignalRadarProvider(frames, base_clock)
        themed = ThemedSignalRadarProvider(frames, themed_clock)

        compared = 0
        for offset in range(10):
            moment = at("2026-02-05") + timedelta(days=offset)
            base_clock.set(moment)
            themed_clock.set(moment)
            for symbol in SYMBOLS:
                expected = base.features(symbol)
                actual = themed.features(symbol)
                for name in SIGNAL_FEATURES:
                    assert actual[name].hex() == expected[name].hex(), (
                        f"{name} cambio en {symbol} el {moment:%Y-%m-%d}"
                    )
                    compared += 1
        assert compared == 10 * len(SYMBOLS) * len(SIGNAL_FEATURES)

    def test_el_radar_base_sigue_publicando_exactamente_seis_claves(self):
        """La subclase es opt-in: quien no la importe no ve un solo numero nuevo."""
        base = SignalRadarProvider(all_frames(), HistoricalClock(at("2026-02-05")))
        assert set(base.features("BTC/USDT")) == set(SIGNAL_FEATURES)

    def test_el_tematico_publica_seis_mas_quince(self):
        themed = ThemedSignalRadarProvider(all_frames(), HistoricalClock(at("2026-02-05")))
        features = themed.features("BTC/USDT")
        assert set(features) == set(SIGNAL_FEATURES) | set(THEME_FEATURES)
        assert len(THEME_FEATURES) == 15
        assert not set(SIGNAL_FEATURES) & set(THEME_FEATURES)

    def test_las_claves_tematicas_siguen_el_orden_canonico(self):
        assert THEME_FEATURES[:3] == theme_features(THEME_NAMES[0])
        for theme in THEME_NAMES:
            tone, intensity, coverage = theme_features(theme)
            assert (tone, intensity, coverage) == (
                f"signal_{theme}_tone",
                f"signal_{theme}_intensity",
                f"signal_{theme}_coverage",
            )


# ------------------------------------------------------------------- la tabla ---------


class TestTabla:
    def test_la_tabla_clasifica_las_treinta(self):
        """Anadir una fuente al catalogo y no darle tema tiene que FALLAR, no desaparecer."""
        themed = {key for spec in THEMES.values() for key in spec.sources}
        excused = {key for key, _ in THEMELESS}
        assert themed | excused == {source.key for source in CATALOG}
        assert len(CATALOG) == 30

    def test_una_fuente_sin_clasificar_revienta_al_validar(self):
        partial = {"flow": ThemeSpec("flow", ("etf_flows",))}
        with pytest.raises(ValueError, match="Unclassified"):
            _check_themes(partial, (), CATALOG)

    def test_ningun_nombre_de_tema_puede_pisar_el_bloque_de_mercado(self):
        """Un tema 'market' generaria `signal_market_tone` y sobrescribiria el bloque del
        radar base al fusionar los diccionarios: es el fallo mas silencioso posible."""
        with pytest.raises(ValueError, match="shadow"):
            _check_themes({"market": ThemeSpec("market", ("etf_flows",))}, (), CATALOG)

    def test_un_tema_no_puede_declarar_una_fuente_inexistente(self):
        with pytest.raises(ValueError, match="absent from the catalog"):
            _check_themes({"x": ThemeSpec("x", ("no_existe",))}, (), CATALOG)

    def test_el_unico_solape_declarado_es_deribit_volatility(self):
        counts: dict[str, int] = {}
        for spec in THEMES.values():
            for key in spec.sources:
                counts[key] = counts.get(key, 0) + 1
        assert sorted(k for k, n in counts.items() if n > 1) == ["deribit_volatility"]
        assert max(counts.values()) <= MAX_THEMES_PER_SOURCE

    def test_todo_tema_declara_por_que_sus_fuentes_estan_juntas(self):
        """La prosa no es decorado: es donde vive el aviso de que el tono de `macro` es ~0."""
        for spec in THEMES.values():
            assert len(spec.reason) > 80


# ----------------------------------------------------------- aritmetica de cobertura ---


class TestCobertura:
    @pytest.mark.parametrize(
        ("theme", "n_sources", "denominator", "min_covered"),
        [
            ("liquidation", 4, 6, 2),
            ("vol_surface", 2, 6, 2),
            ("macro", 6, 6, 2),
            ("attention", 7, 7, 2),
            ("flow", 12, 12, 3),
        ],
    )
    def test_la_aritmetica_por_tema(self, theme, n_sources, denominator, min_covered):
        """Las cinco filas de la tabla del diseno, congeladas."""
        spec = THEMES[theme]
        assert len(spec.sources) == n_sources
        assert effective_denominator(n_sources, spec.min_sources) == denominator
        # La primera cantidad de fuentes cubiertas que abre la puerta es exactamente esta.
        opens = next(k for k in range(1, denominator + 1) if k / denominator >= MIN_SIGNAL_COVERAGE)
        assert opens == min_covered
        assert (min_covered - 1) / denominator < MIN_SIGNAL_COVERAGE

    def test_el_suelo_se_despeja_de_la_propia_constante(self):
        """Seis no es un numero elegido: es (2 - 0,5) / 0,25."""
        assert effective_denominator(1) == 6
        assert effective_denominator(4) == 6
        assert effective_denominator(12) == 12  # por encima del suelo manda lo declarado
        assert MIN_THEME_SOURCES == 2

    def test_un_tema_grande_no_recibe_suelo(self):
        assert effective_denominator(30) == 30

    def test_min_sources_uno_deja_el_suelo_en_dos(self):
        """Es el modo que usa el panel sintetico: un canal unico publica 0,5, no 1,0."""
        assert effective_denominator(1, 1) == 2

    def test_lo_que_se_puede_gatear_en_backtest_historico(self):
        """
        Fragil A PROPOSITO, igual que los tests de `history_from`: cuando una fuente gane
        historia medida, alguien tiene que releer que estrategias dejan de ser ciegas.
        """
        backtestable = {source.key for source in CATALOG if source.backtestable}
        measured = {
            theme: len(set(spec.sources) & backtestable) for theme, spec in THEMES.items()
        }
        assert measured == {
            "liquidation": 1,
            "vol_surface": 1,
            "macro": 3,
            "attention": 2,
            "flow": 8,
        }
        evaluable = {
            theme
            for theme, spec in THEMES.items()
            if measured[theme] / effective_denominator(len(spec.sources), spec.min_sources)
            >= MIN_SIGNAL_COVERAGE
        }
        assert evaluable == {"macro", "attention", "flow"}

    def test_la_cobertura_se_mide_contra_lo_declarado_y_no_contra_lo_cargado(self):
        """Una sola fuente de doce es cobertura 0,08, no 1,0."""
        themed = ThemedSignalRadarProvider(
            {"github_activity": _series(commits=(1.0, 10.0), contributors=(1.0, 4.0))},
            HistoricalClock(at("2026-02-05")),
        )
        features = themed.features("BTC/USDT")
        assert features["signal_flow_coverage"] == pytest.approx(1 / 12)
        assert not theme_reading(features, "flow").readable

    def test_una_sola_fuente_no_abre_ningun_tema(self):
        for theme, spec in THEMES.items():
            denominator = effective_denominator(len(spec.sources), spec.min_sources)
            assert 1 / denominator < MIN_SIGNAL_COVERAGE, theme

    def test_sin_fuentes_los_quince_son_cero_y_ninguno_legible(self):
        themed = ThemedSignalRadarProvider(None, HistoricalClock(at("2026-02-05")))
        features = themed.features("BTC/USDT")
        assert themed.is_empty
        assert all(features[name] == 0.0 for name in THEME_FEATURES)
        assert not any(theme_reading(features, theme).readable for theme in THEME_NAMES)

    def test_una_fuente_rancia_deja_de_cubrir_en_su_tema(self):
        clock = HistoricalClock(at("2026-02-09"))
        frames = {"github_activity": _series(commits=(1.0, 10.0), contributors=(1.0, 4.0))}
        assert (
            ThemedSignalRadarProvider(frames, clock).features("BTC/USDT")["signal_flow_coverage"]
            > 0.0
        )
        clock.set(at("2026-02-09") + timedelta(days=MAX_STALE_DAYS + 5))
        assert (
            ThemedSignalRadarProvider(frames, clock).features("BTC/USDT")["signal_flow_coverage"]
            == 0.0
        )


# --------------------------------------------------------------- rango e invariantes ---


class TestRango:
    def test_rango_declarado_por_tema(self):
        """Sale gratis de `_aggregate`, pero si alguien lo reimplementa hay que enterarse."""
        extreme = {
            "github_activity": _series(commits=(1.0, 1e12), contributors=(1.0, 1e12)),
            "defillama_hacks": frame("*", ["2026-02-01"], hack_amount_usd=[1e15], hack_count=[9.0]),
            "hyperliquid_liqmap": frame(
                "BTC",
                pd.date_range("2026-02-03", periods=2, tz="UTC"),
                liq_cluster_distance_pct=[-0.1, -0.05],
                liq_cluster_notional_usd=[9e12, 9e12],
            ),
        }
        themed = ThemedSignalRadarProvider(extreme, HistoricalClock(at("2026-02-05")))
        features = themed.features("BTC/USDT")
        for theme in THEME_NAMES:
            reading = theme_reading(features, theme)
            assert -Z_CLIP <= reading.tone <= Z_CLIP, theme
            assert 0.0 <= reading.intensity <= Z_CLIP, theme
            assert 0.0 <= reading.coverage <= 1.0, theme

    def test_el_tema_de_solo_mercado_es_identico_para_dos_simbolos(self):
        """El invariante del bloque de mercado, mudado al unico tema 100% de mercado."""
        themed = ThemedSignalRadarProvider(all_frames(), HistoricalClock(at("2026-02-05")))
        btc = themed.features("BTC/USDT")
        eth = themed.features("ETH/USDT")
        assert [btc[n] for n in theme_features("macro")] == [eth[n] for n in theme_features("macro")]

    def test_los_temas_de_activo_difieren_entre_simbolos(self):
        themed = ThemedSignalRadarProvider(all_frames(), HistoricalClock(at("2026-02-05")))
        btc = themed.features("BTC/USDT")
        sol = themed.features("SOL/USDT")
        assert [btc[n] for n in theme_features("attention")] != [
            sol[n] for n in theme_features("attention")
        ]

    def test_un_simbolo_sin_entidad_conserva_el_bloque_de_mercado_del_tema(self):
        """
        Diferencia DECLARADA con `_asset_block`, que devuelve el bloque entero a cero: las
        fuentes de mercado de un tema no dependen del simbolo, asi que no hay motivo para
        apagarlas cuando el simbolo no resuelve.
        """
        themed = ThemedSignalRadarProvider(all_frames(), HistoricalClock(at("2026-02-05")))
        unknown = themed.features("ZZZZ/QQQQ")
        assert all(unknown[name] == 0.0 for name in ASSET_SIGNAL_FEATURES)
        assert unknown["signal_macro_coverage"] > 0.0
        assert unknown["signal_liquidation_coverage"] == 0.0  # tema 100% de activo

    def test_memoiza_el_bloque_tematico_por_el_ahora_del_reloj(self):
        clock = HistoricalClock(at("2026-02-05"))
        themed = ThemedSignalRadarProvider(all_frames(), clock)
        first = themed.features("BTC/USDT")
        assert themed.features("BTC/USDT") == first
        clock.set(at("2026-02-11"))
        assert themed.features("BTC/USDT") is not None

    def test_el_informe_de_cobertura_publica_los_temas(self):
        themed = ThemedSignalRadarProvider(all_frames(), HistoricalClock(at("2026-02-05")))
        report = themed.coverage_report()
        assert set(report["themes"]) == set(THEME_NAMES)
        assert report["themes"]["flow"]["denominator"] == 12
        assert report["min_theme_sources"] == MIN_THEME_SOURCES
        assert "asset_sources" in report  # lo del padre sigue estando


# --------------------------------------------------------------------- la puerta -------


class TestPuerta:
    def test_sin_cobertura_la_puerta_tematica_falla_abierta(self):
        """El invariante central, tema a tema: sin datos no se decide nada."""
        features = dict.fromkeys(THEME_FEATURES, 0.0)
        for theme in THEME_NAMES:
            assert themed_gate_reason(features, theme, min_tone=Z_CLIP) is None
            assert themed_gate_reason(features, theme, min_intensity=Z_CLIP, min_tone=-Z_CLIP) is None

    def test_con_cobertura_la_puerta_si_decide(self):
        features = dict.fromkeys(THEME_FEATURES, 0.0)
        tone, intensity, coverage = theme_features("flow")
        features[coverage] = MIN_SIGNAL_COVERAGE
        features[tone] = -1.0
        features[intensity] = 0.5
        assert themed_gate_reason(features, "flow", min_tone=0.0) is not None
        assert themed_gate_reason(features, "flow", min_tone=-2.0) is None
        assert (
            themed_gate_reason(features, "flow", min_tone=-2.0, max_intensity=0.1) is not None
        )

    def test_justo_por_debajo_del_umbral_no_decide(self):
        features = dict.fromkeys(THEME_FEATURES, 0.0)
        tone, _, coverage = theme_features("flow")
        features[tone] = -Z_CLIP
        features[coverage] = MIN_SIGNAL_COVERAGE - 1e-9
        assert themed_gate_reason(features, "flow", min_tone=Z_CLIP) is None
        features[coverage] = MIN_SIGNAL_COVERAGE
        assert themed_gate_reason(features, "flow", min_tone=Z_CLIP) is not None

    def test_la_puerta_tematica_no_mira_los_bloques_globales(self):
        """Un tema decide con SU cobertura; la del bloque de activo no le sirve de aval."""
        features = dict.fromkeys(list(THEME_FEATURES) + list(SIGNAL_FEATURES), 0.0)
        features["signal_coverage"] = 1.0
        features["signal_tone"] = -3.0
        assert themed_gate_reason(features, "flow", min_tone=0.0) is None

    def test_el_umbral_no_es_configurable_desde_ningun_tema(self):
        for spec in THEMES.values():
            assert spec.min_sources == MIN_THEME_SOURCES
