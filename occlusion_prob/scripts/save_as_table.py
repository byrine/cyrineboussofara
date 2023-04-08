from dataclasses import make_dataclass, dataclass
import pandas as pd


@dataclass
class OcclusionError:
    truth_id: int
    detection_id: int
    frame_nb : int
    stamp: float
    #undetected : str
    merge_or_trunc : str
    occlusion_severity: float
    width_true : float
    height_true: float
    width_det: float
    height_cet: float
    bb_true : list
    bb_det : list




#if __name__ == '__main__':

def table(list_m_t,save_path):
    occlusion_errors = []
    #occlusion_errors.append(OcclusionError(0, 0, 0.0, 0.0))
    #occlusion_errors.append(OcclusionError(1, 2, 0.0, 0.0))

    for i in range(len(list_m_t)):
        a = list_m_t[i]
        for j in range(len(a)):
            b = a[j]
            occlusion_errors.append(OcclusionError(b[0],b[1],b[3],b[2],b[11], b[4], b[7],b[8], b[9],b[10],b[5],b[6]))


    table = pd.DataFrame(occlusion_errors)

    table.to_csv(save_path, sep=';', decimal=',') # I need that to open it with German version of Numbers: sep=';', decimal=','. maybe you do not need it
