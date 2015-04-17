import FunctionTransformer as ft
import numpy as np
import numpy.testing as npt

def test_power_transformer():
    transformer = ft.PowerTransformer(2)
    arr = np.array([-2, 0, 1, 2, 3, 4])
    res = np.array([4, 0, 1, 4, 9, 16])

    npt.assert_array_equal(res, transformer.transform(arr))

def test_power_transformer_zero_power():
    transformer = ft.PowerTransformer(0)
    arr = np.array([-2, 0, 1, 2, 3, 4])
    res = np.array([1, 1, 1, 1, 1, 1])

    npt.assert_array_equal(res, transformer.transform(arr))

def test_log_transformer():
    transformer = ft.LogTransformer()
    arr = np.array([-2, 0, 1, 2, 3, 4])
    res = np.log1p(np.array([-2, 0, 1, 2, 3, 4]))

    npt.assert_array_almost_equal(res, transformer.transform(arr))

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x'],
                   exit=False)
