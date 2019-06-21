import os
import numpy as np
from PIL import Image
class Utils():
    @staticmethod
    def rgb2gray(rgb):
        if type(rgb).__module__ == np.__name__: # numpy type
            return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        elif type(rgb).__module__.startswith("PIL"): #PIL.Image type
            return rgb.convert('L')
        else:
            print("No adjusted type using Utils.rgb2gray()")
            return None
    
    
    @staticmethod
    def rankdata(a):
        """
            returns rank list in decending order
        """
        n = len(a)
        ivec=sorted(range(len(a)), key=a.__getitem__)
        svec=[a[rank] for rank in ivec]
        sumranks = 0
        dupcount = 0
        newarray = [0]*n
        for i in range(n):
            sumranks += i
            dupcount += 1
            if i==n-1 or svec[i] != svec[i+1]:
                averank = sumranks / float(dupcount) + 1
                for j in range(i-dupcount+1,i+1):
                    newarray[ivec[j]] = averank
                sumranks = 0
                dupcount = 0
        return newarray 

    @staticmethod
    def calculate_accuracy(threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

        tpr = 0 if (tp+fn == 0) else float(tp) / float(tp+fn)
        fpr = 0 if (fp+tn == 0) else float(fp) / float(fp+tn) 
        acc = float(tp+tn)/dist.size
        return tpr, fpr, acc


    @staticmethod
    def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, subtract_mean=False): # erase the distance metric
        assert(embeddings1.shape[0] == embeddings2.shape[0])
        assert(embeddings1.shape[1] == embeddings2.shape[1])
        nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
        nrof_thresholds = len(thresholds)
        #TODOTODOTODO
        k_fold = [], []
        for i in range(nrof_folds):
            train_index_list = [i for i in range(i*int(nrof_pairs/nrof_folds), (i+1)*int(nrof_pairs/nrof_folds))]
            test_index_list = [i for i in range(nrof_pairs) if i not in train_index_list]
            #for test_elem in test_index_list:
            #    if test_elem in train_index_list:
            k_fold[0].append(train_index_list)
            k_fold[1].append(test_index_list)
        # k_fold = [i for i in range(0, int(nrof_pairs/nrof_folds))], [i for i in range(int(nrof_pairs/nrof_folds), nrof_pairs)] #equivalent to sklearn.KFold with shuffle=False
        
        tprs = np.zeros((nrof_folds,nrof_thresholds))
        fprs = np.zeros((nrof_folds,nrof_thresholds))
        accuracy = np.zeros((nrof_folds))
        
        indices = np.arange(nrof_pairs)
        
        for fold_idx, (train_set, test_set) in enumerate(zip(k_fold[0], k_fold[1])):
            
            if subtract_mean:
                mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
            else:
                mean = 0.0
            dist = Utils.L2distance(embeddings1-mean, embeddings2-mean)
            
            # Find the best threshold for the fold
            acc_train = np.zeros((nrof_thresholds))
            for threshold_idx, threshold in enumerate(thresholds):
                _, _, acc_train[threshold_idx] = Utils.calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
            best_threshold_index = np.argmax(acc_train)
            for threshold_idx, threshold in enumerate(thresholds):
                tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = Utils.calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
            _, _, accuracy[fold_idx] = Utils.calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
              
            tpr = np.mean(tprs,0)
            fpr = np.mean(fprs,0)
        return tpr, fpr, accuracy
    
    @staticmethod
    def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
        assert(embeddings1.shape[0] == embeddings2.shape[0])
        assert(embeddings1.shape[1] == embeddings2.shape[1])
        nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
        nrof_thresholds = len(thresholds)
        k_fold = [], []
        for i in range(nrof_folds):
            train_index_list = [i for i in range(i*int(nrof_pairs/nrof_folds), (i+1)*int(nrof_pairs/nrof_folds))]
            test_index_list = [i for i in range(nrof_pairs) if i not in train_index_list]
            k_fold[0].append(train_index_list)
            k_fold[1].append(test_index_list)
    
        val = np.zeros(nrof_folds)
        far = np.zeros(nrof_folds)
        
        indices = np.arange(nrof_pairs)
    
        for fold_idx, (train_set, test_set) in enumerate(zip(k_fold[0], k_fold[1])):
            if subtract_mean:
                mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
            else:
                mean = 0.0
            dist = Utils.L2distance(embeddings1-mean, embeddings2-mean)
      
            # Find the threshold that gives FAR = far_target
            far_train = np.zeros(nrof_thresholds)
            for threshold_idx, threshold in enumerate(thresholds):
                _, far_train[threshold_idx] = Utils.calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
            
            if np.max(far_train)>=far_target:
                """
                # previous version for using scipy
                f = interpolate.interp1d(far_train, thresholds, kind='slinear') #jwpyo: interpolate belongs to scipy module
                threshold = f(far_target) # TODO: must resolve this problem
                """
                threshold = np.interp(far_target, far_train, thresholds)
            else:
                threshold = 0.0
        
            val[fold_idx], far[fold_idx] = Utils.calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
      
        val_mean = np.mean(val)
        far_mean = np.mean(far)
        val_std = np.std(val)
        return val_mean, val_std, far_mean
    @staticmethod
    def calculate_val_far(threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
        false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        n_same = np.sum(actual_issame)
        n_diff = np.sum(np.logical_not(actual_issame))
        val = float(true_accept) / float(n_same)
        #TODO solve it!
        far = float(false_accept) / float(n_diff) if n_diff != 0.0 else 0
        return val, far 
    @staticmethod
    def L2distance(v1, v2):
        #TODO solve it!
        assert(len(v1) == len(v2))
        diff = np.subtract(v1, v2)
        dist = np.sum(np.square(diff), 1)
        return dist
        #return np.linalg.norm(np.subtract(v1, v2))
    @staticmethod
    def generate_svg(dwg, text_lines):
        for y, line in enumerate(text_lines):
            dwg.add(dwg.text(line, insert=(11, (y+1)*20+1), fill='white', font_size='20'))
            dwg.add(dwg.text(line, insert=(11, (y+1)*20), fill='white', font_size='20'))
    @staticmethod
    def list_index(seq, item):
        start_at = -1
        locs = []
        while True:
            try:
                loc = seq.index(item, start_at + 1)
            except ValueError:
                break
            else:
                locs.append(loc)
                start_at = loc
        return locs

if __name__ == "__main__":
    #img = Image.open("/home/mendel/Abdoulaye_Wade_0002.png")
    img = Image.open("/home/mendel/parrot.jpg")
