from config import logging
from dataloader.my_dataloader import CustomDataLoader


logger = logging.getLogger(__name__)

class Model:
    def __init__(self, data_path, drop_columns, target_columns):
        self.data = CustomDataLoader(data_path, drop_columns, target_columns)
        self.target_columns = target_columns
        self.minmax_record = {}
        for target_clomuns in target_columns:
            self.minmax_record[target_clomuns] = self.data.data[target_clomuns].min(), self.data.data[target_clomuns].max()

    def inverse_normal_targets(self, target_name, target_value):
        """
        Inverse the normalization of the target value.
        
        Parameters:
            target_name (str): target name
            target_value (float): target value
        
        Returns:
            float: inverse normalized target value
        """
        # df[target_columns] = df[target_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        min_value, max_value = self.minmax_record[target_name]
        return target_value * (max_value - min_value) + min_value