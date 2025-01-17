import pandas as pd
from typing import Callable, Iterable, List
from pandarallel import pandarallel

class Parallelize:

    def __init__(self, operation):
        self.operation = operation

    def __call__(self, input: List) -> List:
        # For some reason parallelize doesn't work. Disable it for now.
        # pandarallel.initialize(progress_bar=False)
        df = pd.DataFrame({"data": input})
        df = df.data.apply(lambda x : self.operation(x))
        return df.to_list()
