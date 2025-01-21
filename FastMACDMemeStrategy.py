from freqtrade.strategy import IStrategy, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from datetime import datetime
from typing import Optional
from freqtrade.persistence import Trade


class FastMACDMemeStrategy(IStrategy):
    """
    优化版MACD策略 - 加入强制止损后的反向开单确认
    """
    INTERFACE_VERSION = 3
    timeframe = '1m'

    # 合约相关设置
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    position_adjustment_enable = True

    # 杠杆设置
    leverage_optimization = False
    initial_leverage = 3

    # 止损设置
    stoploss = -0.12

    # 禁用ROI
    minimal_roi = {
        "0": 100
    }

    # MACD参数
    fast_length = DecimalParameter(2, 10, default=5, space='buy')
    slow_length = DecimalParameter(10, 20, default=12, space='buy')
    signal_length = DecimalParameter(2, 7, default=4, space='buy')

    # 参数设置
    startup_candle_count = 30

    # 记录上一次的强制止损状态
    last_force_exit = False
    last_force_exit_candle = 0
    force_exit_side = None

    # 订单类型设置
    order_types = {
        "entry": "market",
        "exit": "market",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "sucker": "market"
    }

    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 计算MACD
        macd = ta.MACD(dataframe,
                       fastperiod=self.fast_length.value,
                       slowperiod=self.slow_length.value,
                       signalperiod=self.signal_length.value)

        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['macdhist_change'] = dataframe['macdhist'].diff()

        # RSI作为额外确认
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # 做多信号
                    (dataframe['macd'] > dataframe['macdsignal']) &
                    (dataframe['macd'].shift(1) <= dataframe['macdsignal'].shift(1)) &
                    (dataframe['macdhist'] > 0)
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                # 做空信号
                    (dataframe['macd'] < dataframe['macdsignal']) &
                    (dataframe['macd'].shift(1) >= dataframe['macdsignal'].shift(1)) &
                    (dataframe['macdhist'] < 0)
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # 多头平仓
                    (dataframe['macd'] < dataframe['macdsignal']) &
                    (dataframe['macdhist'] < 0)
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                # 空头平仓
                    (dataframe['macd'] > dataframe['macdsignal']) &
                    (dataframe['macdhist'] > 0)
            ),
            'exit_short'] = 1

        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle_idx = len(dataframe) - 1

        # 确定持仓方向
        is_long = trade.amount > 0

        # 计算最大利润
        if hasattr(trade, 'max_rate'):
            if is_long:
                max_profit = (trade.max_rate - trade.open_rate) / trade.open_rate
            else:
                max_profit = (trade.open_rate - trade.max_rate) / trade.open_rate
        else:
            max_profit = current_profit

        # 动态止盈止损设置
        profit_thresholds = {
            0.01: 0.005,  # 盈利1%，止损位0.5%
            0.02: 0.01,  # 盈利2%，止损位1%
            0.03: 0.015,  # 盈利3%，止损位1.5%
            0.05: 0.03,  # 盈利5%，止损位3%
            0.08: 0.05,  # 盈利8%，止损位5%
            0.12: 0.08,  # 盈利12%，止损位8%
            0.15: 0.10,  # 盈利15%，止损位10%
            0.20: 0.15,  # 盈利20%，止损位15%
            0.25: 0.20,  # 盈利25%，止损位20%
            0.30: 0.25  # 盈利30%，止损位25%
        }

        # 紧急止损保护
        if current_profit < -0.02:  # 亏损5%止损
            self.last_force_exit = True
            self.last_force_exit_candle = current_candle_idx
            self.force_exit_side = "long" if is_long else "short"

            if is_long and dataframe['macdhist'].iloc[-1] < 0:
                return '【紧急止损】多单亏损5%且MACD转空'
            elif not is_long and dataframe['macdhist'].iloc[-1] > 0:
                return '【紧急止损】空单亏损5%且MACD转多'

        # 动态止盈止损检查
        for profit_target, stop_loss in sorted(profit_thresholds.items(), reverse=True):
            if current_profit >= profit_target:
                # 结合MACD确认
                if is_long:
                    if ((dataframe['macdhist'].iloc[-1] < 0 and current_profit <= stop_loss) or
                            (dataframe['macdhist'].iloc[-1] < dataframe['macdhist'].iloc[-3] and
                             current_profit <= stop_loss)):
                        return f'【动态止盈】多单达到{profit_target * 100:.2f}%利润，回调至{stop_loss * 100:.2f}%止盈'
                else:
                    if ((dataframe['macdhist'].iloc[-1] > 0 and current_profit <= stop_loss) or
                            (dataframe['macdhist'].iloc[-1] > dataframe['macdhist'].iloc[-3] and
                             current_profit <= stop_loss)):
                        return f'【动态止盈】空单达到{profit_target * 100:.2f}%利润，回调至{stop_loss * 100:.2f}%止盈'
                break

        # 大幅回撤保护
        if max_profit > 0.1:  # 曾经盈利超过10%
            profit_drop = max_profit - current_profit
            if profit_drop > (max_profit * 0.3):  # 回撤超过最大盈利的30%
                return f'【回撤止盈】最大利润{max_profit * 100:.2f}%，回撤幅度{profit_drop * 100:.2f}%'

        # 小额利润保护
        if 0.01 <= current_profit <= 0.02:  # 在1%-2%盈利区间
            if is_long:
                if (dataframe['macdhist'].iloc[-1] < 0 and
                        dataframe['macdhist'].iloc[-1] < dataframe['macdhist'].iloc[-2]):
                    return '【保本止盈】多单1-2%小额利润保护，MACD转弱'
            else:
                if (dataframe['macdhist'].iloc[-1] > 0 and
                        dataframe['macdhist'].iloc[-1] > dataframe['macdhist'].iloc[-2]):
                    return '【保本止盈】空单1-2%小额利润保护，MACD转弱'

        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle_idx = len(dataframe) - 1

        # 检查是否在强制止损后的反向开单
        if self.last_force_exit:
            # 只在止损后的3根K线内检查
            if current_candle_idx - self.last_force_exit_candle <= 3:
                # 如果是之前多单止损，现在准备做空
                if self.force_exit_side == "long" and side == "short":
                    return (dataframe['macd'].iloc[-1] < dataframe['macdsignal'].iloc[-1] and
                            dataframe['macdhist'].iloc[-1] < 0 and
                            dataframe['rsi'].iloc[-1] > 30 and
                            abs(dataframe['macdhist'].iloc[-1]) > abs(dataframe['macdhist'].iloc[-2]))

                # 如果是之前空单止损，现在准备做多
                elif self.force_exit_side == "short" and side == "long":
                    return (dataframe['macd'].iloc[-1] > dataframe['macdsignal'].iloc[-1] and
                            dataframe['macdhist'].iloc[-1] > 0 and
                            dataframe['rsi'].iloc[-1] < 70 and
                            abs(dataframe['macdhist'].iloc[-1]) > abs(dataframe['macdhist'].iloc[-2]))
            else:
                # 超过3根K线后重置状态
                self.last_force_exit = False
                self.force_exit_side = None

        # 常规开单确认
        if side == "long":
            return dataframe['macd'].iloc[-1] > dataframe['macdsignal'].iloc[-1]
        else:
            return dataframe['macd'].iloc[-1] < dataframe['macdsignal'].iloc[-1]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return 3