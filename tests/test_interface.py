# Test the implementation of the interface for each method
import derivative
import inspect
# - Test any error handling of the interface (e.g. invalid method)
# - Test axis


def test_register():
    # Check that every class is registered in methods
    class_list = inspect.getmembers(derivative.dlocal, inspect.isclass)\
                 + inspect.getmembers(derivative.dglobal, inspect.isclass)
    impl_names = [derivative.dlocal.__name__, derivative.dglobal.__name__]
    must_register = [m[0] for m in class_list if m[1].__module__ in impl_names]

    # Interface is registered in each, ignore
    assert len(derivative.methods) == len(must_register)

