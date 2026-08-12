"""
Tests del radar de senales: de diecisiete fuentes a seis numeros, y a una decision.

Lo que hay que blindar aqui no es "que calcule", sino las propiedades sin las cuales el
cableado seria peligroso:

- **La codificacion de evento es distinta de la continua, y lo es EN EL CODIGO.** Una z
  sobre una serie de ceros no falla: da un numero. El test que vale es el que comprueba que
  esa serie NO pasa por ahi.
- **"No hay evento" y "no se de eventos" no son el mismo cero.** Es el fallo que la
  convencion del espacio de observacion (0.0 = neutro) invita a cometer.
- **Ninguna puerta bloquea por falta de datos.** Falla ABIERTA, siempre, y el umbral que lo
  decide no es configurable.
- **Los defaults son inertes POR CONSTRUCCION**, no por ser permisivos: estan en el borde
  exacto del recorte, asi que no existe lectura posible que los active.
- **Mercado y activo estan separados**, y dos simbolos comparten el bloque de mercado.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.observation.signal_radar import (
    ASSET_SIGNAL_FEATURES,
    INERT_MAX_INTENSITY,
    INERT_MIN_INTENSITY,
    INERT_MIN_TONE,
    MARKET_SIGNAL_FEATURES,
    MAX_STALE_DAYS,
    MIN_SIGNAL_COVERAGE,
    POLARITY,
    SIGNAL_FEATURES,
    SignalRadarProvider,
    is_market_scoped,
    signal_gate_reason,
)
from ai_trader.shared.clock import HistoricalClock
from ai_trader.shared.signals import DAY, ENTITY, OBSERVED
from ai_trader.signals.catalog import CATALOG, TIER_MECHANICAL, get_source
from ai_trader.signals.events import (
    DAYS_ACTIVE_CAP,
    DAYS_AHEAD_CAP,
    EVENT_SPECS,
    MAGNITUDE_CLIP,
    SUFFIX_ACTIVE,
    SUFFIX_AHEAD,
    SUFFIX_MAG,
    SUFFIX_SEEN,
    encode_at,
    encode_series,
    event_sources,
    is_event_source,
    pool_report,
)
from ai_trader.signals.normalize import Z_CLIP
from ai_trader.strategies.mean_reversion import MeanReversionConfig, MeanReversionStrategy
from ai_trader.strategies.momentum_crypto import CryptoMomentumConfig, CryptoMomentumStrategy

UTC = timezone.utc


def frame(entity: str, days, **columns) -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [[entity] * len(days), pd.DatetimeIndex(days, tz="UTC")], names=[ENTITY, DAY]
    )
    out = pd.DataFrame(columns, index=index)
    out[OBSERVED] = 1
    return out


def at(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(tzinfo=UTC)


# --------------------------------------------------------- codificacion de evento ----


class TestEventEncoding:
    def _unlocks(self) -> pd.DataFrame:
        return frame(
            "BTC",
            ["2026-01-10", "2026-02-10"],
            unlock_pct_float=[2.0, 5.0],
            unlock_pct_adv=[10.0, 20.0],
        )

    def test_no_hay_evento_a_la_vista_es_un_cero_finito(self):
        """El infinito no entra en un vector de observacion, y las dos salidas perezosas
        mienten: 0 dias diria 'es hoy' y 9999 aplastaria la escala de todo lo demas."""
        encoded = encode_at(self._unlocks(), "token_unlocks", at("2025-01-01"), entities=["BTC"])
        assert encoded["BTC"][f"token_unlocks{SUFFIX_AHEAD}"] == 0.0
        assert all(np.isfinite(v) for v in encoded["BTC"].values())

    def test_la_proximidad_crece_al_acercarse_y_esta_acotada(self):
        events = self._unlocks()
        far = encode_at(events, "token_unlocks", at("2026-01-05"), entities=["BTC"])["BTC"]
        near = encode_at(events, "token_unlocks", at("2026-01-09"), entities=["BTC"])["BTC"]
        assert 0.0 < far[f"token_unlocks{SUFFIX_AHEAD}"] < near[f"token_unlocks{SUFFIX_AHEAD}"]
        assert near[f"token_unlocks{SUFFIX_AHEAD}"] <= 1.0

    def test_justo_en_el_tope_declarado_la_proximidad_se_apaga(self):
        events = frame("BTC", ["2026-02-10"], unlock_pct_float=[1.0], unlock_pct_adv=[1.0])
        day = at("2026-02-10") - timedelta(days=int(DAYS_AHEAD_CAP))
        encoded = encode_at(events, "token_unlocks", day, entities=["BTC"])["BTC"]
        assert encoded[f"token_unlocks{SUFFIX_AHEAD}"] == 0.0
        assert encoded[f"token_unlocks{SUFFIX_SEEN}"] == 1.0  # calendario si hay

    def test_la_estela_decae_y_se_agota_en_su_ventana(self):
        events = self._unlocks()
        just_after = encode_at(events, "token_unlocks", at("2026-01-11"), entities=["BTC"])["BTC"]
        later = encode_at(
            events, "token_unlocks", at("2026-01-10") + timedelta(days=int(DAYS_ACTIVE_CAP)),
            entities=["BTC"],
        )["BTC"]
        assert just_after[f"token_unlocks{SUFFIX_ACTIVE}"] > 0.5
        assert later[f"token_unlocks{SUFFIX_ACTIVE}"] == 0.0

    def test_un_evento_que_no_se_anuncia_no_mira_hacia_adelante(self):
        """La diferencia entre una feature legitima y futuro puro. Un hack no tiene fecha
        conocida, asi que su `_ahead` esta apagado POR CODIGO."""
        hacks = frame("*", ["2026-03-01"], hack_amount_usd=[5e8], hack_count=[1.0])
        before = encode_at(hacks, "defillama_hacks", at("2026-02-28"), entities=["*"])["*"]
        after = encode_at(hacks, "defillama_hacks", at("2026-03-02"), entities=["*"])["*"]
        assert before[f"defillama_hacks{SUFFIX_AHEAD}"] == 0.0
        assert before[f"defillama_hacks{SUFFIX_MAG}"] == 0.0  # no ve el hack de manana
        assert after[f"defillama_hacks{SUFFIX_ACTIVE}"] > 0.0
        assert not EVENT_SPECS["defillama_hacks"].announced

    def test_los_tres_estados_no_se_escriben_igual(self):
        """El corazon del asunto: 'no hay desbloqueo' y 'de este token no se nada' son el
        mismo 0.0 en `_ahead`, y `_seen` es lo unico que los separa."""
        events = self._unlocks()
        sin_evento = encode_at(events, "token_unlocks", at("2025-01-01"), entities=["BTC"])["BTC"]
        sin_datos = encode_at(events, "token_unlocks", at("2026-01-05"), entities=["SEI"])["SEI"]
        assert sin_evento[f"token_unlocks{SUFFIX_AHEAD}"] == 0.0
        assert sin_datos[f"token_unlocks{SUFFIX_AHEAD}"] == 0.0
        assert sin_evento[f"token_unlocks{SUFFIX_SEEN}"] == 1.0
        assert sin_datos[f"token_unlocks{SUFFIX_SEEN}"] == 0.0

    def test_la_magnitud_se_escala_por_su_unidad_declarada_y_se_recorta(self):
        enorme = frame("*", ["2026-03-01"], hack_amount_usd=[1e12], hack_count=[1.0])
        # Al dia siguiente: el dia del evento aun no ha cerrado, como en las barras.
        encoded = encode_at(enorme, "defillama_hacks", at("2026-03-02"), entities=["*"])["*"]
        assert encoded[f"defillama_hacks{SUFFIX_MAG}"] == MAGNITUDE_CLIP

    def test_la_magnitud_conserva_el_signo(self):
        """Un ajuste de dificultad NEGATIVO es mineros apagando: la mitad de la
        informacion esta en el signo, y un valor absoluto la tiraria."""
        caida = frame("bitcoin", ["2026-03-01"], difficulty_change_pct=[-10.0])
        encoded = encode_at(caida, "btc_difficulty", at("2026-03-01"), entities=["bitcoin"])
        assert encoded["bitcoin"][f"btc_difficulty{SUFFIX_MAG}"] < 0

    def test_la_version_densa_coincide_con_la_puntual(self):
        """`encode_series` existe para publicar, y publica lo mismo que decide."""
        events = self._unlocks()
        days = [at("2026-01-05"), at("2026-01-11")]
        dense = encode_series(events, "token_unlocks", days, entities=["BTC"])
        for day in days:
            row = dense.loc[("BTC", pd.Timestamp(day))]
            point = encode_at(events, "token_unlocks", day, entities=["BTC"])["BTC"]
            assert dict(row) == pytest.approx(point)

    def test_la_cadencia_decide_la_codificacion_no_el_tier(self):
        """`staking_queue` es mecanica (tier A) y publica un NIVEL diario. Enrutar por el
        tier le pondria un dias-al-evento a una cola de validadores."""
        assert not is_event_source(get_source("staking_queue"))
        assert get_source("staking_queue").tier == TIER_MECHANICAL
        assert {s.key for s in event_sources()} == set(EVENT_SPECS)
        assert all(s.cadence == "event" for s in event_sources())


class TestPooling:
    def test_el_recuento_es_por_eventos_no_por_entidades(self):
        events = pd.concat(
            [
                frame(token, ["2026-01-10", "2026-02-10"],
                      unlock_pct_float=[1.0, 2.0], unlock_pct_adv=[5.0, 6.0])
                for token in ("BTC", "ETH", "SOL")
            ]
        ).sort_index()
        report = pool_report({"token_unlocks": events})["sources"]["token_unlocks"]
        assert report["pooled_events"] == 6  # 3 tokens x 2 eventos, en la misma distribucion
        assert report["entities"] == 3
        assert report["events_per_entity"] == 2.0

    def test_un_evento_sin_denominador_cuenta_como_evento_pero_no_como_magnitud(self):
        """Son dos muestras distintas: confundirlas es lo que hace creer que hay mas."""
        events = frame("BTC", ["2026-01-10", "2026-02-10"],
                       unlock_pct_float=[1.0, None], unlock_pct_adv=[5.0, 6.0])
        report = pool_report({"token_unlocks": events})["sources"]["token_unlocks"]
        assert report["pooled_events"] == 2
        assert report["with_magnitude"] == 1


# ------------------------------------------------------------------- el radar --------


def _continuous(entities=("BTC", "ETH", "SOL", "ADA", "XRP", "DOGE"), n=40, start="2026-01-01"):
    days = pd.date_range(start, periods=n, tz="UTC")
    return pd.concat(
        [
            frame(e, days, commits=list(np.linspace(1.0, 10.0, n) * (i + 1)),
                  contributors=[float(i + 1)] * n)
            for i, e in enumerate(entities)
        ]
    ).sort_index()


class TestRadar:
    def test_sin_fuentes_el_radar_existe_y_no_dice_nada(self):
        """El estado por defecto del sistema: sin credenciales, sin archivo, sin `[signals]`."""
        radar = SignalRadarProvider(None, HistoricalClock(at("2026-02-01")))
        features = radar.features("BTC/USDT")
        assert radar.is_empty
        assert set(features) == set(SIGNAL_FEATURES)
        assert all(value == 0.0 for value in features.values())

    def test_el_bloque_de_mercado_es_identico_para_todos_los_simbolos(self):
        """El invariante que el generador sintetico va a necesitar, y que separa un factor
        comun de un componente idiosincratico."""
        hacks = frame("*", ["2026-01-20"], hack_amount_usd=[3e8], hack_count=[1.0])
        radar = SignalRadarProvider(
            {"github_activity": _continuous(), "defillama_hacks": hacks},
            HistoricalClock(at("2026-01-25")),
        )
        btc = radar.features("BTC/USDT")
        eth = radar.features("ETH/USDT")
        assert [btc[name] for name in MARKET_SIGNAL_FEATURES] == [
            eth[name] for name in MARKET_SIGNAL_FEATURES
        ]
        assert btc["signal_market_coverage"] > 0.0
        # ...y el de activo no tiene por que serlo (aqui no lo es: distinta actividad).
        assert [btc[name] for name in ASSET_SIGNAL_FEATURES] != [
            eth[name] for name in ASSET_SIGNAL_FEATURES
        ]

    def test_la_separacion_mercado_activo_sale_del_catalogo(self):
        """Es la misma regla que decide a que entidades se pide el dato, no una lista
        paralela que se pueda desincronizar."""
        assert is_market_scoped(get_source("defillama_hacks"))  # alcance de mercado
        assert is_market_scoped(get_source("fred_macro"))  # entidades declaradas
        assert not is_market_scoped(get_source("github_activity"))  # salen del universo

    def test_no_mira_la_barra_de_hoy(self):
        """Misma frontera que las barras y que el regimen: `visible_cutoff`."""
        events = frame("BTC", ["2026-01-20"], unlock_pct_float=[3.0], unlock_pct_adv=[9.0])
        clock = HistoricalClock(at("2026-01-20"))
        radar = SignalRadarProvider({"token_unlocks": events}, clock)
        hoy = radar.features("BTC/USDT")

        clock.set(at("2026-01-21"))
        radar_manana = SignalRadarProvider({"token_unlocks": events}, clock)
        manana = radar_manana.features("BTC/USDT")

        # El dia del evento aun no ha "cerrado": el evento se ve como INMINENTE, no como
        # ocurrido. Al dia siguiente ya es estela.
        assert hoy["signal_tone"] < 0  # unlock a la vista: oferta entrando
        assert manana["signal_tone"] < 0

    def test_memoiza_por_el_ahora_del_reloj(self):
        clock = HistoricalClock(at("2026-02-01"))
        radar = SignalRadarProvider({"github_activity": _continuous()}, clock)
        first = radar.features("BTC/USDT")
        assert radar.features("BTC/USDT") == first
        clock.set(at("2026-02-05"))
        assert radar.features("BTC/USDT") is not None  # se recalcula sin reventar

    def test_una_serie_rancia_deja_de_contar_como_cobertura(self):
        """Una fuente que dejo de publicar seguiria 'cubriendo' con su ultimo valor
        congelado, que es la forma mas silenciosa de que la cobertura mienta."""
        clock = HistoricalClock(at("2026-02-09"))
        radar = SignalRadarProvider({"github_activity": _continuous()}, clock)
        assert radar.features("BTC/USDT")["signal_coverage"] > 0.0

        clock.set(at("2026-02-09") + timedelta(days=MAX_STALE_DAYS + 5))
        radar_viejo = SignalRadarProvider({"github_activity": _continuous()}, clock)
        assert radar_viejo.features("BTC/USDT")["signal_coverage"] == 0.0

    def test_la_cobertura_distingue_al_activo_sin_datos(self):
        events = frame("BTC", ["2026-02-10"], unlock_pct_float=[2.0], unlock_pct_adv=[8.0])
        radar = SignalRadarProvider(
            {"github_activity": _continuous(), "token_unlocks": events},
            HistoricalClock(at("2026-02-05")),
        )
        # Dos fuentes cargadas de las nueve que el catalogo declara por activo: BTC ve las
        # dos y ETH solo una (no tiene calendario de desbloqueos). La proporcion, no el
        # numero, es lo que decide si la puerta llega a evaluarse.
        btc = radar.features("BTC/USDT")["signal_coverage"]
        eth = radar.features("ETH/USDT")["signal_coverage"]
        assert btc == pytest.approx(eth * 2)
        assert 0.0 < eth < btc

    def test_la_cobertura_se_mide_contra_lo_DECLARADO_y_no_contra_lo_cargado(self):
        """Una sola fuente cargada no es cobertura total: es la lectura de una fuente
        haciendose pasar por la del conjunto, que es exactamente lo que el minimo existe
        para impedir."""
        radar = SignalRadarProvider(
            {"github_activity": _continuous()}, HistoricalClock(at("2026-02-05"))
        )
        coverage = radar.features("BTC/USDT")["signal_coverage"]
        assert 0.0 < coverage < MIN_SIGNAL_COVERAGE
        assert signal_gate_reason(radar.features("BTC/USDT"), min_tone=4.0) is None

    def test_el_tono_y_la_intensidad_viven_en_el_rango_declarado(self):
        """Es la propiedad de la que cuelga la inercia de los defaults."""
        extremo = pd.concat(
            [
                frame(e, pd.date_range("2026-01-01", periods=40, tz="UTC"),
                      commits=[1.0] * 39 + [1e12], contributors=[1.0] * 39 + [1e12])
                for e in ("BTC", "ETH", "SOL", "ADA", "XRP", "DOGE")
            ]
        ).sort_index()
        hacks = frame("*", ["2026-02-09"], hack_amount_usd=[1e13], hack_count=[9.0])
        radar = SignalRadarProvider(
            {"github_activity": extremo, "defillama_hacks": hacks},
            HistoricalClock(at("2026-02-10")),
        )
        features = radar.features("BTC/USDT")
        for name in ("signal_tone", "signal_market_tone"):
            assert -Z_CLIP <= features[name] <= Z_CLIP
        for name in ("signal_intensity", "signal_market_intensity"):
            assert 0.0 <= features[name] <= Z_CLIP
        for name in ("signal_coverage", "signal_market_coverage"):
            assert 0.0 <= features[name] <= 1.0

    def test_la_polaridad_es_una_tabla_declarada_sobre_features_que_existen(self):
        catalog_features = {f.name for s in CATALOG for f in s.features}
        event_columns = {f"{key}{SUFFIX_MAG}" for key in EVENT_SPECS}
        for name in POLARITY:
            assert name in catalog_features or name in event_columns, name
        assert set(POLARITY.values()) <= {1.0, -1.0}


# ------------------------------------------------------------------- la puerta -------


class TestGateFailsOpen:
    def _features(self, **overrides) -> dict[str, float]:
        base = dict.fromkeys(SIGNAL_FEATURES, 0.0)
        base.update(overrides)
        return base

    def test_sin_cobertura_no_se_evalua_nada(self):
        """EL INVARIANTE CENTRAL. Un tono catastrofico con cobertura por debajo del minimo
        NO bloquea: falta de datos no es una senal en contra."""
        sin_datos = self._features(signal_tone=-4.0, signal_coverage=0.0)
        assert signal_gate_reason(sin_datos, min_tone=0.0) is None

    def test_con_cobertura_suficiente_la_puerta_si_decide(self):
        con_datos = self._features(signal_tone=-4.0, signal_coverage=1.0)
        assert signal_gate_reason(con_datos, min_tone=0.0) is not None

    def test_cada_bloque_falla_abierto_por_separado(self):
        solo_mercado = self._features(
            signal_tone=-4.0, signal_coverage=0.0,
            signal_market_tone=-4.0, signal_market_coverage=1.0,
        )
        assert "mercado" in signal_gate_reason(solo_mercado, min_tone=0.0)

    def test_el_umbral_de_cobertura_no_es_configurable(self):
        """Si fuera un parametro, un sorteo del optimizador podria subirlo hasta convertir
        el radar en un filtro de disponibilidad de datos."""
        campos = {f.name for f in dataclasses.fields(CryptoMomentumConfig)}
        campos |= {f.name for f in dataclasses.fields(MeanReversionConfig)}
        assert not any("coverage" in name or "cobertura" in name for name in campos)
        assert isinstance(MIN_SIGNAL_COVERAGE, float)

    def test_el_minimo_de_cobertura_es_el_borde_y_no_el_interior(self):
        justo = self._features(signal_tone=-1.0, signal_coverage=MIN_SIGNAL_COVERAGE)
        justo_debajo = self._features(
            signal_tone=-1.0, signal_coverage=MIN_SIGNAL_COVERAGE - 1e-9
        )
        assert signal_gate_reason(justo, min_tone=0.0) is not None
        assert signal_gate_reason(justo_debajo, min_tone=0.0) is None


class TestInertDefaults:
    """
    LOS DEFAULTS NO SON PERMISIVOS: SON IMPOSIBLES DE ACTIVAR.

    Es la mejora concreta sobre `min_relative_strength = -1.0`, que se documento como "sin
    filtro" y no lo era —la fuerza relativa no esta acotada y un -1.0 es alcanzable—. Aqui
    el rango de las lecturas esta cerrado por el recorte de las z, asi que un umbral en el
    borde exacto no puede bloquear ninguna lectura POSIBLE, no solo ninguna probable.
    """

    def _extremes(self):
        for tone in (-Z_CLIP, 0.0, Z_CLIP):
            for intensity in (0.0, Z_CLIP):
                for coverage in (0.0, MIN_SIGNAL_COVERAGE, 1.0):
                    yield {
                        "signal_tone": tone,
                        "signal_intensity": intensity,
                        "signal_coverage": coverage,
                        "signal_market_tone": tone,
                        "signal_market_intensity": intensity,
                        "signal_market_coverage": coverage,
                    }

    def test_los_defaults_estan_en_el_borde_exacto_del_recorte(self):
        assert CryptoMomentumConfig().min_signal_tone == INERT_MIN_TONE == -Z_CLIP
        assert CryptoMomentumConfig().min_signal_intensity == INERT_MIN_INTENSITY == 0.0
        assert MeanReversionConfig().min_signal_tone == INERT_MIN_TONE
        assert MeanReversionConfig().max_signal_intensity == INERT_MAX_INTENSITY == Z_CLIP

    def test_ninguna_lectura_posible_activa_los_defaults(self):
        momentum = CryptoMomentumConfig()
        reversion = MeanReversionConfig()
        for features in self._extremes():
            assert signal_gate_reason(
                features,
                min_tone=momentum.min_signal_tone,
                min_intensity=momentum.min_signal_intensity,
            ) is None
            assert signal_gate_reason(
                features,
                min_tone=reversion.min_signal_tone,
                max_intensity=reversion.max_signal_intensity,
            ) is None

    def test_con_los_defaults_la_puerta_ni_se_consulta(self):
        assert not CryptoMomentumStrategy()._signals_active()
        assert not MeanReversionStrategy()._signals_active()
        assert CryptoMomentumStrategy(
            CryptoMomentumConfig(min_signal_tone=-1.0)
        )._signals_active()
        assert MeanReversionStrategy(
            MeanReversionConfig(max_signal_intensity=2.0)
        )._signals_active()

    def test_un_umbral_fuera_del_rango_se_rechaza_al_construir(self):
        with pytest.raises(ValueError, match="min_signal_tone"):
            CryptoMomentumConfig(min_signal_tone=-10.0)
        with pytest.raises(ValueError, match="min_signal_intensity"):
            CryptoMomentumConfig(min_signal_intensity=99.0)
        with pytest.raises(ValueError, match="max_signal_intensity"):
            MeanReversionConfig(max_signal_intensity=-1.0)


class TestPolarity:
    """
    Momentum y mean-reversion NO quieren lo opuesto de todas las features.

    El tono es un PISO en las dos: en momentum como confirmacion, en reversion como filtro
    de catastrofe (su modo de fallo caracteristico es comprar una caida de -3 sigma que es
    el primer dia de un reprecio permanente). Solo la intensidad es el eje genuinamente
    opuesto: momentum la quiere alta, la reversion baja.
    """

    def test_el_tono_es_un_piso_en_las_dos(self):
        momentum = {f.name for f in dataclasses.fields(CryptoMomentumConfig)}
        reversion = {f.name for f in dataclasses.fields(MeanReversionConfig)}
        assert "min_signal_tone" in momentum
        assert "min_signal_tone" in reversion
        assert "max_signal_tone" not in momentum | reversion

    def test_la_intensidad_es_el_eje_opuesto(self):
        momentum = {f.name for f in dataclasses.fields(CryptoMomentumConfig)}
        reversion = {f.name for f in dataclasses.fields(MeanReversionConfig)}
        assert "min_signal_intensity" in momentum and "max_signal_intensity" not in momentum
        assert "max_signal_intensity" in reversion and "min_signal_intensity" not in reversion

    def test_momentum_pide_confirmacion_y_reversion_huye_del_ruido(self):
        activo = {
            "signal_tone": 1.0,
            "signal_intensity": 3.0,
            "signal_coverage": 1.0,
            **dict.fromkeys(MARKET_SIGNAL_FEATURES, 0.0),
        }
        # La MISMA lectura: intensidad alta con tono a favor.
        assert signal_gate_reason(activo, min_tone=-1.0, min_intensity=2.0) is None
        assert signal_gate_reason(activo, min_tone=-1.0, max_intensity=2.0) is not None

    def test_la_reversion_no_compra_la_catastrofe(self):
        catastrofe = {
            "signal_tone": -3.0,
            "signal_intensity": 3.0,
            "signal_coverage": 1.0,
            **dict.fromkeys(MARKET_SIGNAL_FEATURES, 0.0),
        }
        reason = signal_gate_reason(catastrofe, min_tone=-1.0, max_intensity=Z_CLIP)
        assert reason is not None and "tono" in reason


class TestStrategiesUseTheRadar:
    def _bars(self, closes):
        from tests.conftest import build_bars

        return build_bars(closes)

    def test_momentum_bloquea_con_la_puerta_activa_y_no_sin_ella(self):
        closes = [float(100 + i * 2) for i in range(60)]
        bars = self._bars(closes)

        class Radar:
            def features(self, symbol):
                return {
                    "signal_tone": -2.0,
                    "signal_intensity": 0.0,
                    "signal_coverage": 1.0,
                    **dict.fromkeys(MARKET_SIGNAL_FEATURES, 0.0),
                }

        neutral = CryptoMomentumStrategy()
        neutral.attach_signal_provider(Radar())
        assert neutral.generate_signal("BTC/USDT", bars) is not None

        estricta = CryptoMomentumStrategy(CryptoMomentumConfig(min_signal_tone=0.0))
        estricta.attach_signal_provider(Radar())
        assert estricta.generate_signal("BTC/USDT", bars) is None

    def test_sin_proveedor_la_puerta_no_existe(self):
        closes = [float(100 + i * 2) for i in range(60)]
        estricta = CryptoMomentumStrategy(CryptoMomentumConfig(min_signal_tone=3.9))
        assert estricta.generate_signal("BTC/USDT", self._bars(closes)) is not None

    def test_una_puerta_estricta_sin_cobertura_deja_pasar(self):
        """Falla abierta tambien desde la estrategia, no solo desde el helper."""
        closes = [float(100 + i * 2) for i in range(60)]

        class RadarVacio:
            def features(self, symbol):
                return dict.fromkeys(SIGNAL_FEATURES, 0.0)

        estricta = CryptoMomentumStrategy(CryptoMomentumConfig(min_signal_tone=3.9))
        estricta.attach_signal_provider(RadarVacio())
        assert estricta.generate_signal("BTC/USDT", self._bars(closes)) is not None
