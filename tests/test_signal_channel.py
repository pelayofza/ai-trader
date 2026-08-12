"""
Tests del canal de observacion sintetico y del barrido que produce el break-even.

Lo que hay que blindar aqui no es "que emita numeros", sino las propiedades sin las cuales
el barrido mediria otra cosa que la que dice medir:

- **Los defaults son inertes, y en la direccion correcta.** Un canal recien construido no
  emite NADA. La regla dura de esta pieza es que 0 = MENOS edge: si un default olvidado
  produjera senal perfecta, el estudio entero seria una fabrica de conclusiones falsas.
- **El motor no se entera.** La emision ocurre despues, sobre barras cerradas y con su
  propio generador aleatorio: las velas tienen que salir identicas byte a byte con canal y
  sin el, y los spec.json ya publicados no pueden cambiar ni una clave.
- **El canal entrega lo que declara.** Si se pide rho=0,2 y se entrega 0,05, las celdas del
  barrido estan mal etiquetadas y el break-even es un numero inventado.
- **La causalidad va del mundo a la senal, y solo en esa direccion.** La senal no puede
  correlacionar con retornos YA REALIZADOS, y la estrategia no puede ver la senal de hoy.
- **Se entra por el contrato de produccion.** Mismo `SignalRadarProvider`, misma cobertura,
  misma puerta. Si hubiera un camino paralelo, el barrido no hablaria de lo que opera.
- **El grupo de correlacion distingue multiplicar apuestas de repetir la misma.**
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.observation.signal_radar import (
    ASSET_SIGNAL_FEATURES,
    MARKET_SIGNAL_FEATURES,
    MIN_SIGNAL_COVERAGE,
    POLARITY,
    SIGNAL_FEATURES,
    signal_gate_reason,
)
from ai_trader.scoring.signal_study import (
    ARM_OFF,
    CONFIGS_PER_FAMILY,
    CRITERION,
    GATE_MIN_TONE,
    GATE_PARAM,
    Cell,
    break_even,
    build_cells,
    reproduction_check,
)
from ai_trader.shared.clock import HistoricalClock
from ai_trader.shared.signals import DAY, ENTITY
from ai_trader.signals.catalog import CATALOG
from ai_trader.strategies.momentum_crypto import CryptoMomentumConfig, CryptoMomentumStrategy
from ai_trader.synthetic.engine import PathEngine, ar1_series
from ai_trader.synthetic.fidelity import channel_checks, channel_facts, correlation
from ai_trader.synthetic.scenarios import (
    SIGNAL_CHANNEL_FIELDS,
    FactorPhase,
    ScenarioSpec,
    SignalChannel,
)
from ai_trader.synthetic.signal_channel import (
    SOURCE_PREFIX,
    channel_source,
    channel_values,
    emit_signals,
    feature_name,
    forward_z,
    source_key,
)
from ai_trader.synthetic.universe import CRYPTO, EQUITY, DEFAULT_UNIVERSE

ANCHOR = datetime(2015, 1, 1, tzinfo=timezone.utc)


def _bars(n: int = 400, seed: int = 11, symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")):
    """Barras diarias plausibles (random walk) para varios simbolos."""
    rng = np.random.default_rng(seed)
    index = pd.DatetimeIndex([ANCHOR + timedelta(days=i) for i in range(n)], name="timestamp")
    out = {}
    for i, symbol in enumerate(symbols):
        returns = rng.standard_normal(n) * 0.03
        close = 100.0 * np.exp(np.cumsum(returns))
        out[symbol] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(n, 1_000.0 * (i + 1)),
            },
            index=index,
        )
    return out


def _full_channel(**overrides) -> SignalChannel:
    """Canal con los mandos NO barridos al maximo, que es como los pone el estudio."""
    params = {
        "name": "sweep",
        "rho": 0.2,
        "lead_days": 1,
        "noise_ar": 0.3,
        "informative_share": 1.0,
        "coverage": 1.0,
        "corr_group": "sweep",
    }
    params.update(overrides)
    return SignalChannel(**params)


class TestDefaultsAreInert:
    """0 = MENOS edge, nunca mas. La regla dura de toda la pieza."""

    def test_a_bare_channel_emits_nothing(self):
        # Sin declarar nada: ni un dia informativo, ni un dia observado.
        channel = SignalChannel(name="olvidado")
        assert channel.rho == 0.0
        assert channel.informative_share == 0.0
        assert channel.coverage == 0.0
        assert channel.expected_ic == 0.0

        panel = emit_signals(_bars(), [channel], seed=1)
        emitted = panel.values[("olvidado", "BTC/USDT")]
        assert np.isnan(emitted).all()
        # Sin un solo dia observado no hay ni frame: el radar sale vacio.
        assert panel.frames == {}

    def test_the_negative_name_is_the_dangerous_one(self):
        """`false_positive_rate = 0` seria 'senal perfecta'; por eso el mando va en
        positivo y el default es el otro extremo."""
        assert SignalChannel(name="x").false_positive_rate == 1.0
        assert _full_channel().false_positive_rate == 0.0

    def test_the_default_correlation_group_is_the_least_breadth(self):
        """El default tiene que ser 'la misma apuesta', no 'apuestas independientes': la
        independencia hay que declararla para poder reclamarla."""
        a = SignalChannel(name="a", rho=0.0, coverage=1.0, informative_share=1.0)
        b = SignalChannel(name="b", rho=0.0, coverage=1.0, informative_share=1.0)
        assert a.corr_group == b.corr_group == ""

        panel = emit_signals(_bars(), [a, b], seed=3)
        first = panel.values[("a", "BTC/USDT")]
        second = panel.values[("b", "BTC/USDT")]
        assert correlation(first, second) == pytest.approx(1.0, abs=1e-12)

    def test_an_empty_panel_builds_an_empty_radar(self):
        panel = emit_signals(_bars(), [], seed=1)
        assert panel.is_empty
        radar = panel.provider(HistoricalClock(ANCHOR + timedelta(days=100)))
        assert radar.is_empty
        features = radar.features("BTC/USDT")
        assert features == dict.fromkeys(SIGNAL_FEATURES, 0.0)
        # Y sin cobertura, ninguna puerta puede bloquear.
        assert signal_gate_reason(features, min_tone=4.0) is None


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"rho": 1.5},
            {"rho": -1.5},
            {"lead_days": 0},
            {"noise_ar": 1.0},
            {"informative_share": 1.2},
            {"coverage": -0.1},
            {"name": ""},
        ],
    )
    def test_out_of_range_is_rejected_at_construction(self, kwargs):
        with pytest.raises(ValueError):
            SignalChannel(**{"name": "x", **kwargs})


class TestSerialization:
    def test_a_spec_without_channels_is_byte_identical(self):
        """Los spec.json publicados no pueden ganar una clave. Es la misma promesa que
        sostiene la microestructura de fase."""
        spec = ScenarioSpec(
            id="s", name="S", narrative="",
            phases=(FactorPhase(length_days=10, drift={EQUITY: 0.0}, vol={EQUITY: 0.01}),),
        )
        assert "signals" not in spec.to_dict()
        assert set(spec.to_dict()) == {"id", "name", "narrative", "phases", "shocks", "asset_tilts"}

    def test_only_non_neutral_fields_are_serialized(self):
        assert SignalChannel(name="x").to_dict() == {"name": "x"}
        assert SignalChannel(name="x", rho=0.1).to_dict() == {"name": "x", "rho": 0.1}

    def test_round_trip_through_json(self):
        channel = _full_channel()
        spec = ScenarioSpec(
            id="s", name="S", narrative="n",
            phases=(FactorPhase(length_days=10, drift={}, vol={CRYPTO: 0.02}),),
            signals=(channel,),
        )
        restored = ScenarioSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        assert restored == spec
        assert restored.signals[0] == channel

    def test_every_declared_field_survives_the_round_trip(self):
        """Si alguien anade un mando a SIGNAL_CHANNEL_FIELDS y se olvida del from_dict, el
        canal se leeria con su default -sin edge- y el barrido mediria otra celda."""
        channel = _full_channel(rho=-0.15, lead_days=7, noise_ar=-0.4,
                                informative_share=0.3, coverage=0.6, corr_group="g")
        data = channel.to_dict()
        for field in SIGNAL_CHANNEL_FIELDS:
            assert field in data, f"{field} no se serializa"
        assert SignalChannel.from_dict(data) == channel


class TestEngineIsUntouched:
    """La emision es un PASE APARTE: no puede tocar la secuencia RNG del motor."""

    def _spec(self, **kwargs):
        return ScenarioSpec(
            id="s", name="S", narrative="",
            phases=(
                FactorPhase(length_days=120, drift={EQUITY: 0.0005},
                            vol={EQUITY: 0.008, CRYPTO: 0.02}),
            ),
            **kwargs,
        )

    def test_bars_are_identical_with_and_without_channels(self):
        engine = PathEngine(DEFAULT_UNIVERSE)
        plain = engine.generate(self._spec(), seed=4242)
        with_channels = engine.generate(self._spec(signals=(_full_channel(),)), seed=4242)
        for symbol in plain:
            pd.testing.assert_frame_equal(plain[symbol], with_channels[symbol])

    def test_emitting_between_generations_changes_nothing(self):
        """Aunque el emisor corra EN MEDIO de dos generaciones, las velas no se mueven: su
        generador aleatorio es otro objeto."""
        engine = PathEngine(DEFAULT_UNIVERSE)
        first = engine.generate(self._spec(), seed=99)
        emit_signals(first, [_full_channel()], seed=1)
        second = engine.generate(self._spec(), seed=99)
        for symbol in first:
            pd.testing.assert_frame_equal(first[symbol], second[symbol])

    def test_the_ar1_of_the_noise_is_the_one_of_the_engine(self):
        """El ruido persistente usa la MISMA recurrencia variance-matched que el
        componente idiosincratico, no una copia."""
        eps = np.random.default_rng(0).standard_normal(5_000)
        series = ar1_series(eps, 1.0, np.full(5_000, 0.5))
        assert series.std() == pytest.approx(1.0, abs=0.05)  # variance-matched
        assert correlation(series[:-1], series[1:]) == pytest.approx(0.5, abs=0.05)


class TestDeterminism:
    def test_same_inputs_same_numbers(self):
        bars = _bars()
        first = emit_signals(bars, [_full_channel()], seed=7)
        second = emit_signals(bars, [_full_channel()], seed=7)
        for key, values in first.values.items():
            np.testing.assert_array_equal(values, second.values[key])

    def test_the_seed_moves_the_noise(self):
        bars = _bars()
        a = emit_signals(bars, [_full_channel()], seed=7).values[("sweep", "BTC/USDT")]
        b = emit_signals(bars, [_full_channel()], seed=8).values[("sweep", "BTC/USDT")]
        assert not np.allclose(a, b)

    def test_the_stream_seed_does_not_depend_on_the_process(self):
        """`hash()` de Python esta salado por proceso: con varios workers cada uno emitiria
        un canal distinto. Este valor congelado es lo que lo impide."""
        from ai_trader.synthetic.signal_channel import _stream_seed

        assert _stream_seed(1, "g", "BTC/USDT") == 2054724163262324413


class TestWhatTheChannelDelivers:
    """Si el canal no entrega el rho que declara, las celdas del barrido estan mal
    etiquetadas y el break-even es un numero inventado."""

    @pytest.mark.parametrize("rho", [0.0, 0.05, 0.2, 0.5])
    def test_the_measured_ic_matches_the_declared_one(self, rho):
        bars = _bars(n=3_000, seed=5, symbols=("BTC/USDT",))
        channel = _full_channel(rho=rho)
        panel = emit_signals(bars, [channel], seed=3)
        facts = channel_facts(
            panel.values[("sweep", "BTC/USDT")],
            bars["BTC/USDT"]["close"].to_numpy(float),
            lead_days=channel.lead_days,
        )
        assert facts.ic == pytest.approx(channel.expected_ic, abs=0.04)

    def test_false_positives_dilute_the_ic_proportionally(self):
        bars = _bars(n=3_000, seed=5, symbols=("BTC/USDT",))
        channel = _full_channel(rho=0.4, informative_share=0.5)
        assert channel.expected_ic == pytest.approx(0.2)
        panel = emit_signals(bars, [channel], seed=3)
        facts = channel_facts(
            panel.values[("sweep", "BTC/USDT")],
            bars["BTC/USDT"]["close"].to_numpy(float),
            lead_days=1,
        )
        assert facts.ic == pytest.approx(0.2, abs=0.04)

    def test_a_false_positive_is_indistinguishable_by_size(self):
        """Varianzas igualadas: si los dias no informativos fueran mas pequenos, una
        estrategia podria filtrarlos y el canal seria mejor de lo declarado."""
        bars = _bars(n=4_000, seed=6, symbols=("BTC/USDT",))
        channel = _full_channel(rho=0.6, informative_share=0.5, noise_ar=0.0)
        values = channel_values(
            bars["BTC/USDT"]["close"].to_numpy(float), channel, seed=3, symbol="BTC/USDT"
        )
        assert np.nanstd(values) == pytest.approx(1.0, abs=0.06)

    def test_coverage_sets_the_fraction_of_observed_days(self):
        bars = _bars(n=4_000, seed=7, symbols=("BTC/USDT",))
        channel = _full_channel(coverage=0.4)
        values = channel_values(
            bars["BTC/USDT"]["close"].to_numpy(float), channel, seed=3, symbol="BTC/USDT"
        )
        assert np.isfinite(values).mean() == pytest.approx(0.4, abs=0.03)

    def test_the_noise_is_a_series_and_not_confetti(self):
        bars = _bars(n=4_000, seed=8, symbols=("BTC/USDT",))
        channel = _full_channel(rho=0.0, noise_ar=0.6)
        facts = channel_facts(
            emit_signals(bars, [channel], seed=3).values[("sweep", "BTC/USDT")],
            bars["BTC/USDT"]["close"].to_numpy(float),
            lead_days=1,
        )
        assert facts.ac1 == pytest.approx(0.6, abs=0.05)


class TestCausality:
    """La causalidad va del mundo a la senal. Nunca al reves, y nunca hacia atras."""

    def test_the_signal_does_not_know_the_realized_past(self):
        bars = _bars(n=3_000, seed=9, symbols=("BTC/USDT",))
        facts = channel_facts(
            emit_signals(bars, [_full_channel(rho=0.5)], seed=3).values[("sweep", "BTC/USDT")],
            bars["BTC/USDT"]["close"].to_numpy(float),
            lead_days=1,
        )
        # Con rho=0,5 la correlacion con el retorno que anticipa es enorme...
        assert dict(facts.lead_lag)[0] > 0.4
        # ...y con los que YA ocurrieron, ruido.
        assert facts.past_leak < 0.1

    def test_the_information_sits_where_the_lead_says(self):
        bars = _bars(n=4_000, seed=10, symbols=("BTC/USDT",))
        channel = _full_channel(rho=0.6, lead_days=3)
        facts = channel_facts(
            emit_signals(bars, [channel], seed=3).values[("sweep", "BTC/USDT")],
            bars["BTC/USDT"]["close"].to_numpy(float),
            lead_days=3,
        )
        profile = dict(facts.lead_lag)
        # El retorno de t->t+3 es la suma de los tres dias siguientes: la informacion se
        # reparte entre ellos y no hay nada ni antes ni despues.
        assert all(profile[k] > 0.15 for k in (0, 1, 2))
        assert abs(profile[3]) < 0.1
        assert facts.past_leak < 0.1

    def test_the_last_days_have_no_future_to_observe(self):
        closes = np.linspace(100.0, 200.0, 50)
        z = forward_z(closes, lead_days=5)
        assert np.isnan(z[-5:]).all()
        assert np.isfinite(z[:-5]).all()


class TestCorrelationGroups:
    """Sin este mando no se distingue multiplicar apuestas de repetir la misma."""

    def _pair(self, group_a: str, group_b: str) -> float:
        bars = _bars(n=2_000, seed=12, symbols=("BTC/USDT",))
        channels = [
            SignalChannel(name="a", rho=0.0, coverage=1.0, informative_share=1.0,
                          noise_ar=0.3, corr_group=group_a),
            SignalChannel(name="b", rho=0.0, coverage=1.0, informative_share=1.0,
                          noise_ar=0.3, corr_group=group_b),
        ]
        panel = emit_signals(bars, channels, seed=4)
        return correlation(panel.values[("a", "BTC/USDT")], panel.values[("b", "BTC/USDT")])

    def test_the_same_group_is_the_same_bet(self):
        assert self._pair("uno", "uno") == pytest.approx(1.0, abs=1e-12)

    def test_different_groups_are_independent_bets(self):
        assert abs(self._pair("uno", "dos")) < 0.1

    def test_each_symbol_draws_its_own_noise(self):
        bars = _bars(n=2_000, seed=13)
        panel = emit_signals(bars, [_full_channel(rho=0.0)], seed=5)
        pair = correlation(
            panel.values[("sweep", "BTC/USDT")], panel.values[("sweep", "ETH/USDT")]
        )
        assert abs(pair) < 0.1


class TestProductionContract:
    """Las senales llegan a la estrategia por el MISMO camino que en vivo."""

    def _radar_and_days(self, channel: SignalChannel):
        bars = _bars(n=300, seed=14)
        panel = emit_signals(bars, [channel], seed=6)
        days = bars["BTC/USDT"].index
        clock = HistoricalClock(days[200].to_pydatetime())
        return panel, panel.provider(clock), clock, days, bars

    def test_the_provider_is_the_radar_with_its_six_numbers(self):
        panel, radar, _, _, _ = self._radar_and_days(_full_channel())
        features = radar.features("BTC/USDT")
        assert set(features) == set(SIGNAL_FEATURES)
        # Un canal por activo: cobertura llena en el bloque de activo y cero en el de
        # mercado, que no tiene ninguna fuente declarada.
        assert features["signal_coverage"] == 1.0
        assert features["signal_market_coverage"] == 0.0
        assert all(features[k] == 0.0 for k in MARKET_SIGNAL_FEATURES)
        assert features["signal_tone"] != 0.0

    def test_the_gate_can_block_and_uses_the_asset_block(self):
        _, radar, clock, days, _ = self._radar_and_days(_full_channel())
        blocked = 0
        for day in days[100:250]:
            clock.set(day.to_pydatetime())
            features = radar.features("BTC/USDT")
            assert features["signal_coverage"] >= MIN_SIGNAL_COVERAGE
            if signal_gate_reason(features, min_tone=GATE_MIN_TONE) is not None:
                blocked += 1
        # La puerta corta por la mediana de una z: ni deja pasar todo ni bloquea todo.
        assert 40 < blocked < 110

    def test_the_strategy_never_sees_todays_signal(self):
        """Mismo recorte anti look-ahead que las barras: lo visible es lo ESTRICTAMENTE
        anterior al dia del reloj."""
        panel, radar, clock, days, bars = self._radar_and_days(
            _full_channel(noise_ar=0.0, rho=0.0)
        )
        emitted = panel.values[("sweep", "BTC/USDT")]
        # El tono es la z de la propia serie, monotona en el valor crudo: el signo del tono
        # tiene que seguir al de la senal de AYER, no al de hoy.
        agree_yesterday = agree_today = 0
        for i in range(120, 260):
            clock.set(days[i].to_pydatetime())
            tone = radar.features("BTC/USDT")["signal_tone"]
            if tone == 0.0:
                continue
            agree_yesterday += int(np.sign(tone) == np.sign(emitted[i - 1]))
            agree_today += int(np.sign(tone) == np.sign(emitted[i]))
        assert agree_yesterday > agree_today

    def test_the_gate_reaches_a_real_strategy(self):
        """El contrato completo: `attach_signal_provider` + `signal_gate_reason`.

        Se comprueba sobre una ventana que SI produce senal con la configuracion inerte,
        porque si no el test pasaria por el motivo equivocado (una estrategia que no opera
        tampoco opera con la puerta cerrada)."""
        rng = np.random.default_rng(21)
        n = 300
        closes = 100.0 * np.exp(np.cumsum(0.006 + rng.standard_normal(n) * 0.01))
        index = pd.DatetimeIndex(
            [ANCHOR + timedelta(days=i) for i in range(n)], name="timestamp"
        )
        window = pd.DataFrame(
            {"open": closes, "high": closes * 1.02, "low": closes * 0.98,
             "close": closes, "volume": np.full(n, 1_000.0)},
            index=index,
        )
        bars = {"BTC/USDT": window, "ETH/USDT": window * 1.5}
        panel = emit_signals(bars, [_full_channel()], seed=6)
        clock = HistoricalClock(index[-1].to_pydatetime())
        radar = panel.provider(clock)

        inert = CryptoMomentumStrategy(CryptoMomentumConfig(require_breakout=False))
        inert.attach_signal_provider(radar)
        assert not inert._signals_active()  # el default no puede filtrar nada
        assert inert.generate_signal("BTC/USDT", window) is not None

        # Un piso de tono en el extremo del recorte: ninguna lectura posible lo supera.
        strict = CryptoMomentumStrategy(
            CryptoMomentumConfig(require_breakout=False, min_signal_tone=3.9)
        )
        strict.attach_signal_provider(radar)
        assert strict.generate_signal("BTC/USDT", window) is None

    def test_simulated_sources_never_enter_the_real_catalog(self):
        source = channel_source(_full_channel())
        assert source.key.startswith(SOURCE_PREFIX)
        assert source.key not in {s.key for s in CATALOG}
        assert all(not s.key.startswith(SOURCE_PREFIX) for s in CATALOG)
        # Y su polaridad se INYECTA, no se escribe en la tabla de las fuentes reales.
        assert feature_name(_full_channel()) not in POLARITY

    def test_symbols_without_entity_are_declared_not_dropped_in_silence(self):
        bars = _bars(n=100, symbols=("BTC/USDT",))
        bars["/USDT"] = bars["BTC/USDT"]
        panel = emit_signals(bars, [_full_channel()], seed=1)
        assert dict(panel.omitted).get("/USDT")
        assert ("sweep", "/USDT") not in panel.values

    def test_the_frame_is_the_canonical_shape_of_the_ingest(self):
        panel = emit_signals(_bars(n=200), [_full_channel()], seed=1)
        frame = panel.frames[source_key(_full_channel())]
        assert list(frame.index.names) == [ENTITY, DAY]
        assert feature_name(_full_channel()) in frame.columns
        assert set(frame.index.get_level_values(ENTITY)) == {"BTC", "ETH"}


class TestSweepReading:
    """El criterio de lectura esta declarado ANTES de correr, y se aplica tal cual."""

    def test_the_grid_always_carries_the_control_and_the_off_cell(self):
        cells = build_cells((0.0, 0.1), (1,))
        assert [c.cell_id for c in cells] == [ARM_OFF, "rho0_h1", "rho0.1_h1"]
        assert cells[0].channel is None
        assert cells[1].channel.rho == 0.0  # el control existe siempre

    def test_only_the_gate_threshold_is_injected(self):
        from ai_trader.scoring.transfer_study import build_specs

        spec = build_specs()[0]
        cell = Cell(cell_id="c", arm="on", rho=0.1, lead_days=1)
        injected = cell.spec_for(spec)
        assert injected.id == spec.id  # los config_id son LITERALMENTE los publicados
        assert injected.params[GATE_PARAM] == GATE_MIN_TONE
        assert {k: v for k, v in injected.params.items() if k != GATE_PARAM} == spec.params
        # Y la celda `off` es la configuracion publicada, sin tocar.
        assert Cell(cell_id=ARM_OFF, arm=ARM_OFF).spec_for(spec) is spec

    def _verdict(self, rho, beats, **extra):
        return {
            "arm": "on", "rho": rho, "lead_days": 1, "beats": beats,
            "margin": 0.1 if beats else -0.1, "selected": "c#00",
            "selected_reward_validation": 1.0 if beats else -1.0, **extra,
        }

    def test_a_dirty_control_voids_the_sweep(self):
        """Si rho=0 bate al baseline, lo que se ha medido no es informacion."""
        result = break_even([self._verdict(0.0, True), self._verdict(0.1, True)])
        assert result["control_clean"] is False
        assert result["verdict"] == "anulado_por_el_control"

    def test_the_break_even_is_the_smallest_rho_that_beats(self):
        result = break_even([
            self._verdict(0.0, False),
            self._verdict(0.05, False),
            self._verdict(0.1, True),
            self._verdict(0.2, True),
        ])
        assert result["control_clean"] is True
        assert result["verdict"] == "break_even_encontrado"
        assert result["by_lead"][0]["break_even_rho"] == 0.1

    def test_not_reaching_it_is_a_result_and_says_how_far(self):
        result = break_even([self._verdict(0.0, False), self._verdict(0.2, False)])
        assert result["verdict"] == "no_alcanzado_en_la_rejilla"
        assert result["by_lead"][0]["break_even_rho"] is None
        assert result["by_lead"][0]["break_even_above"] == 0.2

    def test_the_criterion_is_declared_in_code(self):
        for key in ("seleccion", "lectura", "batir", "break_even", "control", "delta"):
            assert CRITERION[key]

    def test_reproduction_is_refused_when_the_grid_is_not_the_published_one(self):
        """Mismo `config_id` con otro tamano de hipercubo NO es la misma configuracion."""
        plan = dataclasses.replace(_fake_plan(), configs_per_family=2)
        check = reproduction_check(plan, [])
        assert check["available"] is False
        assert "hipercubo" in check["reason"]

        available = reproduction_check(
            dataclasses.replace(_fake_plan(), configs_per_family=CONFIGS_PER_FAMILY),
            [],
            units_path="no/existe.json",
        )
        assert available["available"] is False
        assert "no esta" in available["reason"]


def _fake_plan():
    from ai_trader.scoring.signal_study import StudyPlan

    return StudyPlan(
        library_id="ai_v3", config_path="config/default.toml", symbols=("BTC/USDT",),
        library_symbols_omitted=(), scenario_ids=("a", "b"), train_scenarios=("a",),
        validation_scenarios=("b",), split_seed=1, n_paths=1, start="2015-01-01T00:00:00+00:00",
        end="2016-01-01T00:00:00+00:00", window_days=365, cells=(), families=(),
        configs_per_family=CONFIGS_PER_FAMILY, study_seed=1, emission_seed=1,
        scheme="cpcv", n_groups=6, n_test_groups=2, n_folds=15, purge_days=10,
        periods_per_year=365, cvar_alpha=0.25, starting_equity=10_000.0,
    )


class TestChannelAcceptance:
    """El loop de calibracion legitimo, extendido a las senales: se cierra sobre
    PROPIEDADES DEL MUNDO —IC, autocorrelacion, lead-lag— y no sobre el rendimiento de
    ninguna estrategia. Y puede FALLAR, que es lo que lo hace evidencia."""

    def test_a_channel_that_delivers_passes(self):
        checks = channel_checks({"a": 0.10}, {"a": {"ic": 0.104, "past_leak": 0.03}})
        assert all(c.passed for c in checks)

    def test_a_channel_that_lies_about_its_ic_fails(self):
        checks = channel_checks({"a": 0.10}, {"a": {"ic": 0.02, "past_leak": 0.03}})
        assert not checks[0].passed

    def test_a_channel_that_reads_the_past_fails(self):
        checks = channel_checks({"a": 0.10}, {"a": {"ic": 0.10, "past_leak": 0.4}})
        assert not checks[1].passed

    def test_declared_but_not_measured_is_not_approved(self):
        """No medir no es aprobar: es el mismo criterio que el resto del estudio."""
        assert not any(c.passed for c in channel_checks({"a": 0.1}, {}))

    def test_the_published_libraries_declare_no_channels(self):
        """Hoy ninguna libreria publicada trae canales, asi que el estudio de fidelidad no
        lee un solo parquet de mas por esto."""
        spec = ScenarioSpec(
            id="s", name="S", narrative="",
            phases=(FactorPhase(length_days=10, drift={}, vol={CRYPTO: 0.02}),),
        )
        assert spec.signals == ()


class TestAssetBlockIsTheOneThatMoves:
    def test_two_symbols_get_different_signals(self):
        """El canal es por ACTIVO: si dos simbolos vieran lo mismo, seria una senal de
        mercado y el bloque que hay que mirar seria el otro."""
        panel = emit_signals(_bars(n=300, seed=15), [_full_channel()], seed=8)
        clock = HistoricalClock((ANCHOR + timedelta(days=200)))
        radar = panel.provider(clock)
        btc = radar.features("BTC/USDT")
        eth = radar.features("ETH/USDT")
        assert btc["signal_tone"] != eth["signal_tone"]
        # ...y el bloque de mercado sigue siendo identico (vacio) para los dos.
        assert all(btc[k] == eth[k] for k in MARKET_SIGNAL_FEATURES)
        assert set(ASSET_SIGNAL_FEATURES).issubset(btc)
