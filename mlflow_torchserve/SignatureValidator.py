import numpy as np
import pandas
from mlflow.exceptions import MlflowException
from mlflow.types import DataType, Schema, TensorSpec
from mlflow.types.utils import clean_tensor_type


class SignatureValidator:
    def __init__(self, model_meta):
        self._model_meta = model_meta

    @property
    def metadata(self):
        """Model metadata."""
        if self._model_meta is None:
            raise MlflowException("Model is missing metadata.")
        return self._model_meta

    def _enforce_mlflow_datatype(self, name, values: pandas.Series, t: DataType):
        """
        Enforce the input column type matches the declared in model input schema.

        The following type conversions are allowed:

        1. np.object -> string
        2. int -> long (upcast)
        3. float -> double (upcast)
        4. int -> double (safe conversion)

        Any other type mismatch will raise error.
        """
        if values.dtype == np.object and t not in (DataType.binary, DataType.string):
            values = values.infer_objects()

        if t == DataType.string and values.dtype == np.object:
            #  NB: strings are by default parsed and inferred as objects, but it is
            # recommended to use StringDtype extension type if available. See
            #
            # `https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html`
            #
            # for more detail.
            try:
                return values.astype(t.to_pandas(), errors="raise")
            except ValueError:
                raise MlflowException(
                    "Failed to convert column {0} from type {1} to {2}.".format(
                        name, values.dtype, t
                    )
                )

        # NB: Comparison of pandas and numpy data type fails
        # when numpy data type is on the left hand
        # side of the comparison operator.
        # It works, however, if pandas type is on the left hand side.
        # That is because pandas is aware of numpy.
        if t.to_pandas() == values.dtype or t.to_numpy() == values.dtype:
            # The types are already compatible => conversion is not necessary.
            return values

        if t == DataType.binary and values.dtype.kind == t.binary.to_numpy().kind:
            # NB: bytes in numpy have variable itemsize depending on the length of the longest
            # element in the array (column). Since MLflow binary type is length agnostic, we ignore
            # itemsize when matching binary columns.
            return values

        numpy_type = t.to_numpy()
        if values.dtype.kind == numpy_type.kind:
            is_upcast = values.dtype.itemsize <= numpy_type.itemsize
        elif values.dtype.kind == "u" and numpy_type.kind == "i":
            is_upcast = values.dtype.itemsize < numpy_type.itemsize
        elif values.dtype.kind in ("i", "u") and numpy_type == np.float64:
            # allow (u)int => double conversion
            is_upcast = values.dtype.itemsize <= 6
        else:
            is_upcast = False

        if is_upcast:
            return values.astype(numpy_type, errors="raise")
        else:
            # NB: conversion between incompatible types (e.g. floats -> ints or
            # double -> float) are not allowed. While supported by pandas and numpy,
            # these conversions alter the values significantly.
            def all_ints(xs):
                return all([pandas.isnull(x) or int(x) == x for x in xs])

            hint = ""
            if (
                values.dtype == np.float64
                and numpy_type.kind in ("i", "u")
                and values.hasnans
                and all_ints(values)
            ):
                hint = (
                    " Hint: the type mismatch is likely caused by missing values. "
                    "Integer columns in python can not represent missing values and are therefore "
                    "encoded as floats. The best way to avoid this problem is to infer the model "
                    "schema based on a realistic data sample"
                    " (training dataset) that includes missing "
                    "values. Alternatively, you can declare integer columns as doubles (float64) "
                    "whenever these columns may have missing values. See `Handling Integers With "
                    "Missing Values <https://www.mlflow.org/docs/latest/models.html#"
                    "handling-integers-with-missing-values>`_ for more details."
                )

            raise MlflowException(
                "Incompatible input types for column {0}. "
                "Can not safely convert {1} to {2}.{3}".format(name, values.dtype, numpy_type, hint)
            )

    def _enforce_tensor_spec(self, values: np.ndarray, tensor_spec: TensorSpec):
        """
        Enforce the input tensor shape and type matches the provided tensor spec.
        """
        expected_shape = tensor_spec.shape
        actual_shape = values.shape
        if len(expected_shape) != len(actual_shape):
            raise MlflowException(
                "Shape of input {0} does not match expected shape {1}.".format(
                    actual_shape, expected_shape
                )
            )
        for expected, actual in zip(expected_shape, actual_shape):
            if expected == -1:
                continue
            if expected != actual:
                raise MlflowException(
                    "Shape of input {0} does not match expected shape {1}.".format(
                        actual_shape, expected_shape
                    )
                )
        if clean_tensor_type(values.dtype) != tensor_spec.type:
            raise MlflowException(
                "dtype of input {0} does not match expected dtype {1}".format(
                    values.dtype, tensor_spec.type
                )
            )
        return values

    def _enforce_col_schema(self, pfInput, input_schema: Schema):
        """Enforce the input columns conform to the model's column-based signature."""
        if input_schema.has_input_names():
            input_names = input_schema.input_names()
        else:
            input_names = pfInput.columns[: len(input_schema.inputs)]
        input_types = input_schema.input_types()
        new_pfInput = pandas.DataFrame()
        for i, x in enumerate(input_names):
            new_pfInput[x] = self._enforce_mlflow_datatype(x, pfInput[x], input_types[i])
        return new_pfInput

    def _enforce_tensor_schema(self, pfInput, input_schema: Schema):
        """Enforce the input tensor(s) conforms to the model's tensor-based signature."""
        if input_schema.has_input_names():
            if isinstance(pfInput, dict):
                new_pfInput = dict()
                for col_name, tensor_spec in zip(input_schema.input_names(), input_schema.inputs):
                    if not isinstance(pfInput[col_name], np.ndarray):
                        raise MlflowException(
                            "This model contains a tensor-based model signature with input names,"
                            " which suggests a dictionary input mapping input name to a numpy"
                            " array, but a dict with value type {0} was found.".format(
                                type(pfInput[col_name])
                            )
                        )
                    new_pfInput[col_name] = self._enforce_tensor_spec(
                        pfInput[col_name], tensor_spec
                    )
            elif isinstance(pfInput, pandas.DataFrame):
                new_pfInput = dict()
                for col_name, tensor_spec in zip(input_schema.input_names(), input_schema.inputs):
                    new_pfInput[col_name] = self._enforce_tensor_spec(
                        np.array(pfInput[col_name], dtype=tensor_spec.type), tensor_spec
                    )
            else:
                raise MlflowException(
                    "This model contains a tensor-based model signature with input names, which"
                    " suggests a dictionary input mapping input name to tensor, but an input of"
                    " type {0} was found.".format(type(pfInput))
                )
        else:
            if isinstance(pfInput, pandas.DataFrame):
                new_pfInput = self._enforce_tensor_spec(pfInput.to_numpy(), input_schema.inputs[0])
            elif isinstance(pfInput, np.ndarray):
                new_pfInput = self._enforce_tensor_spec(pfInput, input_schema.inputs[0])
            else:
                raise MlflowException(
                    "This model contains a tensor-based model signature with no input names,"
                    " which suggests a numpy array input, but an input of type {0} was"
                    " found.".format(type(pfInput))
                )
        return new_pfInput

    def _enforce_schema(self, pfInput, input_schema: Schema):
        """
        Enforces the provided input matches the model's input schema,

        For signatures with input names, we check there are no missing inputs and reorder
        the inputs to match the ordering declared in schema if necessary.
        Any extra columns are ignored.

        For column-based signatures, we make sure the types of the input match the type specified in
        the schema or if it can be safely converted to match the input schema.

        For tensor-based signatures, we make sure the shape and type of the input matches the shape
        and type specified in model's input schema.
        """
        if not input_schema.is_tensor_spec():
            if isinstance(pfInput, (list, np.ndarray, dict)):
                try:
                    pfInput = pandas.DataFrame(pfInput)
                except Exception as e:
                    raise MlflowException(
                        "This model contains a column-based signature, which suggests a DataFrame"
                        " input. There was an error casting the input data to a DataFrame:"
                        " {0}".format(str(e))
                    )
            if not isinstance(pfInput, pandas.DataFrame):
                raise MlflowException(
                    "Expected input to be DataFrame or list. Found: %s" % type(pfInput).__name__
                )

        if input_schema.has_input_names():
            # make sure there are no missing columns
            input_names = input_schema.input_names()
            expected_cols = set(input_names)
            actual_cols = set()
            if len(expected_cols) == 1 and isinstance(pfInput, np.ndarray):
                # for schemas with a single column, match input with column
                pfInput = {input_names[0]: pfInput}
                actual_cols = expected_cols
            elif isinstance(pfInput, pandas.DataFrame):
                actual_cols = set(pfInput.columns)
            elif isinstance(pfInput, dict):
                actual_cols = set(pfInput.keys())
            missing_cols = expected_cols - actual_cols
            extra_cols = actual_cols - expected_cols
            # Preserve order from the original columns, since missing/extra columns are likely to
            # be in same order.
            missing_cols = [c for c in input_names if c in missing_cols]
            extra_cols = [c for c in actual_cols if c in extra_cols]
            if missing_cols:
                raise MlflowException(
                    "Model is missing inputs {0}."
                    " Note that there were extra inputs: {1}".format(missing_cols, extra_cols)
                )
        elif not input_schema.is_tensor_spec():
            # The model signature does not specify column names => we can only verify column count.
            num_actual_columns = len(pfInput.columns)
            if num_actual_columns < len(input_schema.inputs):
                raise MlflowException(
                    "Model inference is missing inputs. The model signature declares "
                    "{0} inputs  but the provided value only has "
                    "{1} inputs. Note: the inputs were not named in the signature so we can "
                    "only verify their count.".format(len(input_schema.inputs), num_actual_columns)
                )

        return (
            self._enforce_tensor_schema(pfInput, input_schema)
            if input_schema.is_tensor_spec()
            else self._enforce_col_schema(pfInput, input_schema)
        )
