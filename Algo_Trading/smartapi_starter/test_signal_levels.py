from strategy import calculate_trade_levels


def test_buy_levels():
    levels = calculate_trade_levels(100.0, "BUY")
    assert levels["current_price"] == 100.0
    assert levels["target_price"] == 102.0
    assert levels["stop_loss"] == 98.0


def test_sell_levels():
    levels = calculate_trade_levels(100.0, "SELL")
    assert levels["target_price"] == 98.0
    assert levels["stop_loss"] == 102.0
