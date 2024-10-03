import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from typing import Dict, List

class HSIEvaluation:
    def __init__(self, param: Dict):
        self.param = param
        self.target_names = self.get_target_names()
        self.res = {}

    def get_target_names(self) -> List[str]:
        data_sign = self.param['data']['data_sign']
        if data_sign == 'IndianPine':  
            return ['Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees', 
                     'Hay-windrowed', 'Soybean-notill', 'Soybean-mintill',
                    'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers','Alfalfa','Grass-pasture-mowed', 'Oats']
        elif data_sign == "Pavia":
            return ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted_metal_sheets', 'Bare_Soil', 
                    'Bitumen', 'Self_Blocking_Bricks', 'Shadows']
        elif data_sign == "Houston":
            return ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees', 'Soil', 'Water', 
                    'Residential', 'Commercial', 'Road', 'Highway', 'Railway', 'Parking Lot 1', 
                    'Parking Lot 2', 'Tennis Court', 'Running Track']
        elif data_sign == "Plastic":
            return ['Fillet', 'ABS', 'FAB', 'HDPE', 'LDPE', 'NYL', 'PET', 'PP', 'PS', 'PUR', 'PVC', 'RUB', 'TEF']
        else:
            return None

    def AA_andEachClassAccuracy(self, confusion_matrix: np.ndarray) -> tuple:
        list_diag = np.diag(confusion_matrix)
        list_raw_sum = np.sum(confusion_matrix, axis=1)
        each_acc = np.nan_to_num(list_diag / list_raw_sum)
        average_acc = np.mean(each_acc)
        return each_acc, average_acc

    def eval(self, y_test: np.ndarray, y_pred_test: np.ndarray) -> Dict:
        class_num = np.max(y_test) + 1
        classification = classification_report(y_test, y_pred_test, 
                labels=list(range(class_num)), digits=4, target_names=self.target_names, zero_division=0)
        oa = accuracy_score(y_test, y_pred_test)
        confusion = confusion_matrix(y_test, y_pred_test)
        each_acc, aa = self.AA_andEachClassAccuracy(confusion)
        kappa = cohen_kappa_score(y_test, y_pred_test)

        self.res['classification'] = str(classification)
        self.res['oa'] = oa * 100
        self.res['confusion'] = str(confusion)
        self.res['each_acc'] = str(each_acc * 100)
        self.res['aa'] = aa * 100
        self.res['kappa'] = kappa * 100
        return self.res