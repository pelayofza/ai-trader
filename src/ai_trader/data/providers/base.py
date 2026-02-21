from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd


class MarketDataProvider(ABC):

    @abstractmethod
    def get_daily_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame | None:
        """
        Return a DataFrame with columns:
        [Open, High, Low, Close, Volume]
        Indexed by datetime.
        """
        pass