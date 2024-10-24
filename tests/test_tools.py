import unittest 
import pandas as pd
import numpy as np

from ctgan_tools import DataSampler, DataPrepare

class TestTools(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
                'Names':['Ande',"Nick","Moris",'Ande'],
                'Age':[10, 20, 30, 25],
                'Weight':[50,80, 120,95]
                })
        


    def test_check_feature_cat_values(self):
        transformer = DataPrepare()
        data = transformer.transform(self.df, ['Names'])
        sampler = DataSampler(data, transformer)
        result = sampler.feature_cat_values

        resolve =[
                    [ [0,1],[2,3]],
                    [[0,3],[2],[1]],
                    [[1,3],[2],[0]]
                ]
        
        self.assertEqual(len(resolve), len(result)), "не совпали размерности в результате"

        assert len(resolve) == len(result), "не совпало кол-во признаков в feature_cat_values"

        for col in range(len(result)):
            assert len(result[col]) == len(resolve[col]), f"не совпала размерность для признака по номеру {col}"
            for cat in range(len(result[col])):
                assert list(result[col][cat]) == resolve[col][cat], f"не совпала разметка для [{col}, {cat}] "


        # assert np.array(resolve) == np.array(result), "Результат не совпал с проверкой feature_cat_values"


if __name__ == '__main__':
    unittest.main()