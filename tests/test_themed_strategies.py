"""
Tests de las seis primitivas tematicas.

Lo que hay que blindar, por orden de importancia:

- **Los defaults son inertes POR CONSTRUCCION, y aqui inerte es mas fuerte que en momentum**:
  con la config por defecto la capa no puede cambiar la elegibilidad, ni el lado, ni el
  tamano, CON NINGUN radar —vacio o lleno—. Se comprueba por producto cartesiano de lecturas
  extremas, no con un caso.
- **Radar vacio da EXACTAMENTE la misma senal que sin proveedor.** Es la propiedad que hace
  que anadir la capa no cambie una sola cifra de lo ya publicado.
- **Sin cobertura degrada a la variante ciega**, incluso con la capa completamente armada.
  Es la limitacion declarada del ranking historico, convertida en test.
- **Con cobertura la capa SI cambia lado, tamano y elegibilidad.** El espejo del anterior: si
  no cambiara nada nunca, la capa seria decorativa y los tests de inercia pasarian igual.
- **Un corto viaja entero por el runner.** El motor soporta cortos de punta a punta pero no
  habia un solo test que abriera uno; cinco de estas seis los emiten.
"""
from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest

from ai_trader.execution.paper import PaperExecutionConfig, PaperExecutionEngine
from ai_trader.execution.router import ExecutionRouter
from ai_trader.observation.signal_radar import (
    INERT_MAX_INTENSITY,
    INERT_MIN_INTENSITY,
    INERT_MIN_TONE,
    MIN_SIGNAL_COVERAGE,
    SIGNAL_FEATURES,
)
from ai_trader.observation.signal_themes import (
    THEME_FEATURES,
    THEME_NAMES,
    theme_features,
)
from ai_trader.risk.engine import RiskEngine
from ai_trader.shared.instruments import AssetClass
from ai_trader.shared.schemas import PositionStatus, Side
from ai_trader.strategies.registry import STRATEGY_REGISTRY, build_strategy
from ai_trader.strategies.signal_layer import (
    INERT_SIGNAL_WEIGHT,
    SIDE_CORE,
    SIDE_TONE,
    SIDE_VETO,
)
from tests.conftest import build_bars

# Las seis familias nuevas y el tema que lee cada una.
THEMED_FAMILIES = {
    "liquidation_cascade": "liquidation",
    "vol_term_structure": "vol_surface",
    "event_calendar_drift": "macro",
    "attention_ignition": "attention",
    "flow_persistence": "flow",
    "signal_composite": "composite",
}

# Las que operan los dos lados. `attention_ignition` es solo largo POR TESIS.
TWO_SIDED = tuple(f for f in THEMED_FAMILIES if f != "attention_ignition")


class FakeThemedProvider:
    """
    Un radar de mentira con la forma exacta del de verdad: 6 + 15 claves.

    Se le dicta la terna de cada tema. No hereda del proveedor real a proposito: lo que las
    estrategias consumen es un contrato de duck typing (`features(symbol) -> dict`), y un
    doble que lo cumpla es la unica forma de comprobar que no dependen de nada mas.
    """

    def __init__(self, **themes: tuple[float, float, float]) -> None:
        self._features = dict.fromkeys(list(SIGNAL_FEATURES) + list(THEME_FEATURES), 0.0)
        for theme, (tone, intensity, coverage) in themes.items():
            tone_key, intensity_key, coverage_key = theme_features(theme)
            self._features[tone_key] = tone
            self._features[intensity_key] = intensity
            self._features[coverage_key] = coverage

    def features(self, symbol: str) -> dict[str, float]:
        return dict(self._features)


def covered(theme: str, tone: float, intensity: float = 0.5) -> FakeThemedProvider:
    """Un radar en el que SOLO `theme` tiene cobertura suficiente para decidir."""
    return FakeThemedProvider(**{theme: (tone, intensity, 1.0)})


def all_covered(tone: float, intensity: float = 0.5) -> FakeThemedProvider:
    """Los cinco temas cubiertos: lo que necesita el compuesto para ser legible."""
    return FakeThemedProvider(**{name: (tone, intensity, 1.0) for name in THEME_NAMES})


def provider_for(family: str, tone: float, intensity: float = 0.5) -> FakeThemedProvider:
    theme = THEMED_FAMILIES[family]
    return all_covered(tone, intensity) if theme == "composite" else covered(theme, tone, intensity)


# ------------------------------------------------------------ los patrones de precio ---
#
# Una serie por familia que SI dispara, y su negativo. Construidas a mano y no muestreadas:
# un test que solo comprueba "no revienta" no distingue una primitiva de una funcion que
# devuelve None.


def capitulation_bars(*, down: bool = True):
    """Rango muy por encima del ATR y cierre pegado al extremo del dia."""
    flat = [100.0 + 0.4 * np.sin(i / 3) for i in range(59)]
    if down:
        closes = flat + [88.0]
        highs = [c + 0.6 for c in flat] + [99.0]
        lows = [c - 0.6 for c in flat] + [87.0]
    else:
        closes = flat + [112.0]
        highs = [c + 0.6 for c in flat] + [113.0]
        lows = [c - 0.6 for c in flat] + [101.0]
    return build_bars(closes, highs=highs, lows=lows)


def compression_breakout_bars(*, up: bool = True):
    """Setenta dias agitados, cuarenta en calma y una rotura del canal."""
    rng = np.random.default_rng(3)
    noisy = list(100 + np.cumsum(rng.normal(0, 2.0, 70)))
    calm = list(noisy[-1] + np.cumsum(rng.normal(0, 0.10, 40)))
    edge = max(calm[-25:]) + 2.5 if up else min(calm[-25:]) - 2.5
    return build_bars(noisy + calm + [edge])


def drift_bars(*, up: bool = True):
    """Plano y luego una deriva de cinco dias dentro de la banda."""
    step = 0.8 if up else -0.8
    return build_bars([100.0] * 45 + [100.0 + step * i for i in range(1, 6)])


def ignition_bars(*, loud: bool = True):
    """Tendencia larga y una barra con volumen multiplo cerrando en maximos."""
    trend = list(np.linspace(80, 120, 79))
    closes = trend + [126.0]
    highs = [c + 1.0 for c in trend] + [126.3]
    lows = [c - 1.0 for c in trend] + [121.0]
    volumes = [1_000.0] * 79 + [4_000.0 if loud else 1_000.0]
    return build_bars(closes, highs=highs, lows=lows, volumes=volumes)


def pullback_bars(*, up: bool = True):
    """Tendencia persistente con un retroceso que vuelve a tocar la media."""
    if up:
        return build_bars(list(np.linspace(100, 118, 95)) + [117.0, 115.5, 114.2])
    return build_bars(list(np.linspace(118, 100, 95)) + [101.0, 102.5, 103.8])


def cross_bars(*, up: bool = True):
    """Tendencia larga, un hueco por debajo de la media corta y el cruce de vuelta."""
    if up:
        base = list(np.linspace(100, 200, 140))
        return build_bars(base + [base[-1] - 8, base[-1] - 9, base[-1] - 6, base[-1] + 1])
    base = list(np.linspace(200, 100, 140))
    return build_bars(base + [base[-1] + 8, base[-1] + 9, base[-1] + 6, base[-1] - 1])


FIRING_BARS = {
    "liquidation_cascade": capitulation_bars(),
    "vol_term_structure": compression_breakout_bars(),
    "event_calendar_drift": drift_bars(),
    "attention_ignition": ignition_bars(),
    "flow_persistence": pullback_bars(),
    "signal_composite": cross_bars(),
}

SHORT_BARS = {
    "liquidation_cascade": capitulation_bars(down=False),
    "vol_term_structure": compression_breakout_bars(up=False),
    "event_calendar_drift": drift_bars(up=False),
    "flow_persistence": pullback_bars(up=False),
    "signal_composite": cross_bars(up=False),
}

# Negativos: la misma familia sobre una serie que NO cumple su patron.
QUIET_BARS = {
    "liquidation_cascade": build_bars([100.0] * 60),
    "vol_term_structure": build_bars(list(np.linspace(100, 130, 111))),
    "event_calendar_drift": build_bars([100.0] * 50),
    "attention_ignition": ignition_bars(loud=False),
    "flow_persistence": build_bars([100.0] * 98),
    "signal_composite": build_bars([100.0] * 144),
}


# ------------------------------------------------------------------- el registro -------


class TestRegistro:
    def test_las_seis_estan_en_el_registro_y_en_el_espacio(self):
        from ai_trader.scoring.search_space import SPACES

        for family in THEMED_FAMILIES:
            assert family in STRATEGY_REGISTRY
            assert family in SPACES

    def test_cada_familia_declara_su_tema(self):
        for family, theme in THEMED_FAMILIES.items():
            assert build_strategy(family).theme == theme

    def test_ninguna_soporta_simbolos_de_prediccion(self):
        for family in THEMED_FAMILIES:
            strategy = build_strategy(family)
            assert not strategy.supports_symbol("PM::algo")
            assert strategy.supports_symbol("BTC/USDT")

    @pytest.mark.parametrize("family", sorted(THEMED_FAMILIES))
    def test_cada_familia_dispara_en_su_patron_y_calla_en_el_negativo(self, family):
        assert build_strategy(family).generate_signal("BTC/USDT", FIRING_BARS[family]) is not None
        assert build_strategy(family).generate_signal("BTC/USDT", QUIET_BARS[family]) is None

    @pytest.mark.parametrize("family", sorted(TWO_SIDED))
    def test_las_de_dos_lados_emiten_corto_en_el_patron_espejo(self, family):
        signal = build_strategy(family).generate_signal("BTC/USDT", SHORT_BARS[family])
        assert signal is not None and signal.side is Side.SELL
        assert signal.stop_loss > signal.entry_price > signal.take_profit

    def test_la_ignicion_de_atencion_no_puede_emitir_un_corto(self):
        """`allow_short = False` es la tesis, no un default: la atencion llega comprando."""
        assert build_strategy("attention_ignition").config.allow_short is False
        for bars in (ignition_bars(), capitulation_bars(down=False)):
            signal = build_strategy("attention_ignition").generate_signal("BTC/USDT", bars)
            assert signal is None or signal.side is Side.BUY

    @pytest.mark.parametrize("family", sorted(THEMED_FAMILIES))
    def test_una_serie_corta_o_vacia_no_revienta(self, family):
        import pandas as pd

        strategy = build_strategy(family)
        assert strategy.generate_signal("BTC/USDT", pd.DataFrame()) is None
        assert strategy.generate_signal("BTC/USDT", build_bars([100.0] * 5)) is None


# --------------------------------------------------------------------- la inercia ------


class TestInercia:
    @pytest.mark.parametrize("family", sorted(THEMED_FAMILIES))
    def test_los_defaults_estan_en_el_borde_exacto_de_su_rango(self, family):
        config = build_strategy(family).config
        fields = {f.name for f in dataclasses.fields(config)}
        if "min_signal_tone" in fields:
            assert config.min_signal_tone == INERT_MIN_TONE
        if "min_signal_intensity" in fields:
            assert config.min_signal_intensity == INERT_MIN_INTENSITY
        if "max_signal_intensity" in fields:
            assert config.max_signal_intensity == INERT_MAX_INTENSITY
        if "signal_side_mode" in fields:
            assert config.signal_side_mode == SIDE_CORE
        assert config.signal_weight == INERT_SIGNAL_WEIGHT

    def test_el_tema_sin_direccion_no_declara_tono_ni_modo_de_lado(self):
        """
        `event_calendar_drift` NO tiene `min_signal_tone` ni `signal_side_mode`, y es una
        ausencia deliberada: de las seis fuentes de `macro` solo `ofac_sdn` tiene polaridad,
        asi que el tono del tema es ~0 por construccion. Un umbral de tono ahi seria un mando
        que parece hacer algo y no puede.
        """
        fields = {f.name for f in dataclasses.fields(build_strategy("event_calendar_drift").config)}
        assert "min_signal_tone" not in fields
        assert "signal_side_mode" not in fields
        assert "min_signal_intensity" in fields and "max_signal_intensity" in fields

    @pytest.mark.parametrize("family", sorted(THEMED_FAMILIES))
    def test_con_los_defaults_la_capa_ni_se_consulta(self, family):
        assert build_strategy(family)._signals_active() is False

    @pytest.mark.parametrize("family", sorted(THEMED_FAMILIES))
    def test_mover_cualquier_mando_activa_la_capa(self, family):
        fields = {f.name for f in dataclasses.fields(build_strategy(family).config)}
        knobs = {"signal_weight": 0.5}
        if "min_signal_tone" in fields:
            knobs["min_signal_tone"] = 0.0
        if "min_signal_intensity" in fields:
            knobs["min_signal_intensity"] = 0.5
        if "max_signal_intensity" in fields:
            knobs["max_signal_intensity"] = 1.0
        if "signal_side_mode" in fields:
            knobs["signal_side_mode"] = SIDE_VETO
        for name, value in knobs.items():
            assert build_strategy(family, {name: value})._signals_active() is True, name

    @pytest.mark.parametrize("family", sorted(THEMED_FAMILIES))
    def test_ninguna_lectura_posible_activa_los_defaults(self, family):
        """
        Producto cartesiano de lecturas extremas contra la config por defecto. Si alguna
        combinacion cambiara la senal, "inerte" seria una promesa y no una propiedad.
        """
        bars = FIRING_BARS[family]
        expected = build_strategy(family).generate_signal("BTC/USDT", bars)
        assert expected is not None

        extremes = itertools.product(
            (-4.0, 0.0, 4.0), (0.0, 4.0), (0.0, MIN_SIGNAL_COVERAGE, 1.0)
        )
        for tone, intensity, coverage in extremes:
            strategy = build_strategy(family)
            strategy.attach_signal_provider(
                FakeThemedProvider(**{n: (tone, intensity, coverage) for n in THEME_NAMES})
            )
            actual = strategy.generate_signal("BTC/USDT", bars)
            assert actual is not None
            assert (actual.side, actual.confidence, actual.entry_price) == (
                expected.side, expected.confidence, expected.entry_price
            )

    @pytest.mark.parametrize("family", sorted(THEMED_FAMILIES))
    def test_radar_vacio_da_exactamente_la_misma_senal_que_el_nucleo(self, family):
        """La propiedad que hace que anadir la capa no mueva una sola cifra publicada."""
        from ai_trader.observation.signal_themes import ThemedSignalRadarProvider
        from ai_trader.shared.clock import HistoricalClock
        from datetime import datetime, timezone

        bars = FIRING_BARS[family]
        blind = build_strategy(family).generate_signal("BTC/USDT", bars)

        wired = build_strategy(family)
        wired.attach_signal_provider(
            ThemedSignalRadarProvider(
                None, HistoricalClock(datetime(2026, 3, 1, tzinfo=timezone.utc))
            )
        )
        with_empty_radar = wired.generate_signal("BTC/USDT", bars)

        assert blind is not None and with_empty_radar is not None
        for field in ("side", "confidence", "entry_price", "stop_loss", "take_profit", "reason"):
            assert getattr(blind, field) == getattr(with_empty_radar, field), field
        # Sin lectura, `Signal.features` no inventa tres ceros: no hay claves de tema.
        assert not any(k.startswith("signal_theme_") for k in with_empty_radar.features)


# ------------------------------------------------------- degradacion y activacion ------


class TestDegradacion:
    @pytest.mark.parametrize("family", sorted(THEMED_FAMILIES))
    def test_sin_cobertura_degrada_a_la_variante_ciega(self, family):
        """
        La capa COMPLETAMENTE armada, y aun asi la senal es la del nucleo ciego: sin
        cobertura no se decide nada. Es la limitacion del ranking historico, como test.
        """
        bars = FIRING_BARS[family]
        blind = build_strategy(family).generate_signal("BTC/USDT", bars)
        assert blind is not None

        fields = {f.name for f in dataclasses.fields(build_strategy(family).config)}
        armed = {"signal_weight": 1.0}
        if "min_signal_tone" in fields:
            armed["min_signal_tone"] = -INERT_MIN_TONE  # el techo: bloquearia todo
        if "min_signal_intensity" in fields:
            armed["min_signal_intensity"] = INERT_MAX_INTENSITY
        if "signal_side_mode" in fields:
            armed["signal_side_mode"] = SIDE_TONE

        strategy = build_strategy(family, armed)
        # Una sola fuente por tema, muy por debajo del minimo: exactamente el caso historico.
        strategy.attach_signal_provider(
            FakeThemedProvider(**{n: (-4.0, 0.0, MIN_SIGNAL_COVERAGE / 2) for n in THEME_NAMES})
        )
        actual = strategy.generate_signal("BTC/USDT", bars)
        assert actual is not None
        assert (actual.side, actual.confidence) == (blind.side, blind.confidence)

    @pytest.mark.parametrize("family", sorted(f for f in THEMED_FAMILIES if f != "event_calendar_drift"))
    def test_con_cobertura_un_tono_en_contra_bloquea(self, family):
        strategy = build_strategy(family, {"min_signal_tone": 0.0})
        strategy.attach_signal_provider(provider_for(family, tone=-3.0))
        assert strategy.generate_signal("BTC/USDT", FIRING_BARS[family]) is None

    @pytest.mark.parametrize("family", sorted(f for f in THEMED_FAMILIES if f != "event_calendar_drift"))
    def test_con_cobertura_un_tono_a_favor_deja_pasar(self, family):
        strategy = build_strategy(family, {"min_signal_tone": 0.0})
        strategy.attach_signal_provider(provider_for(family, tone=3.0))
        signal = strategy.generate_signal("BTC/USDT", FIRING_BARS[family])
        assert signal is not None
        assert signal.features["signal_theme_coverage"] >= MIN_SIGNAL_COVERAGE

    def test_el_calendario_bloquea_por_intensidad_y_no_por_tono(self):
        family = "event_calendar_drift"
        low = build_strategy(family, {"min_signal_intensity": 2.0})
        low.attach_signal_provider(covered("macro", tone=0.0, intensity=0.1))
        assert low.generate_signal("BTC/USDT", FIRING_BARS[family]) is None

        high = build_strategy(family, {"max_signal_intensity": 0.5})
        high.attach_signal_provider(covered("macro", tone=0.0, intensity=3.0))
        assert high.generate_signal("BTC/USDT", FIRING_BARS[family]) is None

        ok = build_strategy(family, {"min_signal_intensity": 0.5, "max_signal_intensity": 3.0})
        ok.attach_signal_provider(covered("macro", tone=-4.0, intensity=1.0))
        assert ok.generate_signal("BTC/USDT", FIRING_BARS[family]) is not None

    @pytest.mark.parametrize(
        "family", sorted(f for f in TWO_SIDED if f != "event_calendar_drift")
    )
    def test_el_veto_cancela_pero_nunca_invierte(self, family):
        strategy = build_strategy(family, {"signal_side_mode": SIDE_VETO,
                                           "signal_tone_threshold": 0.5})
        # Tono muy en contra del lado que propone el nucleo (largo en todos los patrones).
        strategy.attach_signal_provider(provider_for(family, tone=-3.0))
        assert strategy.generate_signal("BTC/USDT", FIRING_BARS[family]) is None

    def test_el_tono_puede_invertir_el_lado_en_modo_tone(self):
        """`flow` es el unico tema con tono de calidad, y `SIDE_TONE` es su modo natural."""
        strategy = build_strategy(
            "flow_persistence", {"signal_side_mode": SIDE_TONE, "signal_tone_threshold": 0.5}
        )
        strategy.attach_signal_provider(covered("flow", tone=-3.0))
        signal = strategy.generate_signal("BTC/USDT", FIRING_BARS["flow_persistence"])
        assert signal is not None and signal.side is Side.SELL
        assert signal.stop_loss > signal.entry_price > signal.take_profit

    def test_el_peso_mueve_el_tamano_y_solo_si_hay_cobertura(self):
        bars = FIRING_BARS["flow_persistence"]
        base = build_strategy("flow_persistence").generate_signal("BTC/USDT", bars)

        favour = build_strategy("flow_persistence", {"signal_weight": 1.0})
        favour.attach_signal_provider(covered("flow", tone=4.0))
        against = build_strategy("flow_persistence", {"signal_weight": 1.0})
        against.attach_signal_provider(covered("flow", tone=-4.0))
        blind = build_strategy("flow_persistence", {"signal_weight": 1.0})
        blind.attach_signal_provider(covered("flow", tone=4.0, intensity=0.0))
        blind._signals.__init__(flow=(4.0, 0.0, MIN_SIGNAL_COVERAGE / 2))

        assert favour.generate_signal("BTC/USDT", bars).confidence > base.confidence
        assert against.generate_signal("BTC/USDT", bars).confidence < base.confidence
        assert blind.generate_signal("BTC/USDT", bars).confidence == base.confidence

    def test_el_compuesto_necesita_dos_temas_legibles(self):
        """
        La regla del minimo de dos sale de la aritmetica que ya existia: 1/5 = 0,20 < 0,25.
        Con un solo tema cubierto el compuesto NO es legible y la capa no decide nada.
        """
        bars = FIRING_BARS["signal_composite"]
        blind = build_strategy("signal_composite").generate_signal("BTC/USDT", bars)

        one = build_strategy("signal_composite", {"min_signal_tone": 0.0})
        one.attach_signal_provider(covered("flow", tone=-4.0))
        assert one.generate_signal("BTC/USDT", bars).confidence == blind.confidence

        two = build_strategy("signal_composite", {"min_signal_tone": 0.0})
        two.attach_signal_provider(FakeThemedProvider(flow=(-4.0, 1.0, 1.0),
                                                      macro=(-4.0, 1.0, 1.0)))
        assert two.generate_signal("BTC/USDT", bars) is None

    def test_el_compuesto_promedia_solo_los_temas_legibles(self):
        """Un tema sin cobertura no es un tema neutro: es uno del que no se sabe nada, y
        meterlo como cero diluiria a los que si dicen algo."""
        bars = FIRING_BARS["signal_composite"]
        strategy = build_strategy("signal_composite", {"signal_weight": 0.5})
        strategy.attach_signal_provider(
            FakeThemedProvider(flow=(3.0, 1.0, 1.0), macro=(1.0, 1.0, 1.0), attention=(9.9, 9.9, 0.0))
        )
        signal = strategy.generate_signal("BTC/USDT", bars)
        # (3 + 1) / 2 = 2, ignorando el tema con cobertura 0 aunque grite.
        assert signal.features["signal_theme_tone"] == pytest.approx(2.0)
        assert signal.features["signal_theme_coverage"] == pytest.approx(2 / 5)


# ------------------------------------------------------------------- validacion --------


class TestValidacion:
    @pytest.mark.parametrize("family", sorted(THEMED_FAMILIES))
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("min_signal_tone", -5.0),
            ("min_signal_tone", 5.0),
            ("min_signal_intensity", -0.1),
            ("min_signal_intensity", 5.0),
            ("max_signal_intensity", -0.1),
            ("max_signal_intensity", 5.0),
            ("signal_side_mode", "lo_que_sea"),
            ("signal_weight", -0.1),
            ("signal_weight", 2.0),
        ],
    )
    def test_un_umbral_fuera_de_rango_se_rechaza_al_construir(self, family, field, value):
        fields = {f.name for f in dataclasses.fields(build_strategy(family).config)}
        if field not in fields:
            pytest.skip(f"{family} no declara {field}")
        with pytest.raises(ValueError):
            build_strategy(family, {field: value})

    def test_una_ventana_rapida_mas_larga_que_la_lenta_se_rechaza(self):
        """Se RECHAZA en vez de repararse: el cociente mediria lo contrario en silencio."""
        with pytest.raises(ValueError, match="rv_slow_window"):
            build_strategy("vol_term_structure", {"rv_fast_window": 60, "rv_slow_window": 30})

    def test_una_persistencia_por_debajo_de_la_mitad_se_rechaza(self):
        with pytest.raises(ValueError, match="min_persistence"):
            build_strategy("flow_persistence", {"min_persistence": 0.3})

    def test_un_extremo_por_encima_del_centro_se_rechaza(self):
        """`close_location_max > 0.5` invertiria la primitiva en silencio."""
        with pytest.raises(ValueError, match="close_location_max"):
            build_strategy("liquidation_cascade", {"close_location_max": 0.7})


# ----------------------------------------------------- el corto, de punta a punta ------


class TestCortoEndToEnd:
    """
    El camino que hoy no cubre ningun test: `Position`, el modelo intrabarra, el riesgo y el
    motor de papel son todos simetricos, pero nadie habia abierto un corto por el runner.
    Cinco de las seis primitivas nuevas los emiten, asi que deja de ser hipotetico.
    """

    def test_un_corto_viaja_entero_por_el_runner(self, tmp_path, limits):
        from ai_trader.app.runner import RunnerConfig, TradingRunner
        from ai_trader.app.state_store import JsonStateStore
        from ai_trader.execution.polymarket_paper import PolymarketPaperExecutionEngine
        from test_runner import FakeMarketData

        bars = SHORT_BARS["flow_persistence"]
        strategy = build_strategy("flow_persistence")
        signal = strategy.generate_signal("BTC/USDT", bars)
        assert signal is not None and signal.side is Side.SELL

        paper = PaperExecutionEngine(PaperExecutionConfig(fee_rate=0.001, slippage_bps=5.0))
        runner = TradingRunner(
            config=RunnerConfig(symbols=["BTC/USDT"], symbol_cooldown_hours=0),
            market_data_reader=FakeMarketData(bars={"BTC/USDT": bars}),
            strategies=[strategy],
            risk_engine=RiskEngine(limits),
            execution_router=ExecutionRouter.paper(
                spot_engine=paper,
                prediction_engine=PolymarketPaperExecutionEngine(paper_engine=paper),
            ),
            state_store=JsonStateStore(tmp_path / "state.json"),
        )

        runner.run_cycle()
        positions = runner.get_positions()
        assert len(positions) == 1
        position = positions[0]
        assert position.side is Side.SELL
        assert position.status is PositionStatus.OPEN
        assert position.stop_loss > position.entry_price > position.take_profit
        # El PnL de un corto sube cuando el precio baja. Es lo que hace que el lado exista.
        assert position.gross_pnl_at(position.entry_price * 0.9) > 0
        assert position.gross_pnl_at(position.entry_price * 1.1) < 0
        assert position.asset_class is AssetClass.CRYPTO
