from src.performance_tracking import get_performance_summary


def test_performance_summary_returns_dict():
    result = get_performance_summary()
    assert isinstance(result, dict)
