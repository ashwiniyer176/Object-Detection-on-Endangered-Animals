import numpy as np

class Digit:
    
    def __init__(self, results):
        """initialise the class. in this the results get sorted and stats of the box get recorded as per the function.

        Args:
            results (dict): results as returned by the model
        """
        
        self.results = results
        self.stats = {}
        self.sort_results()
        self.box_stats()
    
    def sort_results(self,threshold=0):
        """sort the model according to the xmin (or x-axis) coordinate

        Args:
            threshold (int, optional): threshold, if any, below which the box is not to be considered. Defaults to 0.
        """

        def get_elem(arr):
            """return the xmin coordinate of a given box
            """
            return arr[1]

        boxes, score = self.results['detection_boxes'],self.results['detection_scores']
        boxes = boxes[score>=threshold]
        score = score[score>=threshold]
        # print("initial:\n",boxes)
        boxes = sorted(boxes, key=get_elem)
        self.results = {
            "detection_boxes" : boxes,
            "detection_scores" : score
        }
        # print("final:\n",boxes)
        # return np.array(boxes)
    
    def box_stats(self):
        """calculate statistical values of the given box (used for outlier detection). as of now the following things
           are calculated -
           1. width and height of each bb(bounding box).
           2. mean and std of width and height
           3. tolerance of width and height
        """
        ymin = [arr[0] for arr in self.results['detection_boxes']]
        xmin = [arr[1] for arr in self.results['detection_boxes']]
        ymax = [arr[2] for arr in self.results['detection_boxes']]
        xmax = [arr[3] for arr in self.results['detection_boxes']]

        width = [int(max)-int(min) for max,min in zip(xmax,xmin)]
        height = [int(max)-int(min) for max,min in zip(ymax,ymin)]

        ## Calculate mean width

        mean_width = np.mean(width)
        std_width = np.std(width)

        width_tolerance = [mean_width-2*std_width, mean_width+2*std_width]

        ## Calculate mean height

        mean_height = np.mean(height)
        std_height = np.std(height)

        height_tolerance = [mean_height-2*std_height, mean_height+2*std_height]

        self.stats['width_tol'] = width_tolerance
        self.stats['height_tol'] = height_tolerance
        self.stats['width_mean'] = mean_width
        self.stats['height_mean'] = mean_height
    
    def remove_outlier(self):
        """remove boxes which are not in the range of width and height tolerance (mean +- 2*standard deviation)

        Returns:
            results: removed outlier results
        """

        new_results = []
        new_score = []
        boxes, scores = self.results['detection_boxes'],self.results['detection_scores']

        # print(self.stats['width_tol'],self.stats['height_tol'])

        for cnt,box in enumerate(boxes):
            ymin,xmin,ymax,xmax = box
            width = xmax-xmin
            height = ymax-ymin

            if width<self.stats['width_tol'][0] or width>self.stats['width_tol'][1]:
                continue
            elif height<self.stats['height_tol'][0] or height>self.stats['height_tol'][1]:
                continue
            else:
                new_results.append(box)
                new_score.append(scores[cnt])
        
        self.results['detection_boxes'] = new_results
        self.results['detection_scores'] = new_score
        return self.results


    def remove_concurrent(self, relax_factor=5):
        """remove boxes which overlap. here the relax factor is for how much two boxes overlap.

        Args:
            relax_factor (int, optional): tolerance of overlap. Defaults to 5.

        Returns:
            results: removed overlapped boxes
        """

        new_boxes = []
        new_scores = []
        boxes = self.results['detection_boxes']
        scores = self.results['detection_scores']
        skip_index = []

        for index in range(len(boxes)):
            # print(skip_index)

            if index in skip_index:
                continue

            for index2 in range(index+1,len(boxes)):

                pt_max = boxes[index][3]
                pt_min = boxes[index2][1]

                if pt_max-pt_min<relax_factor:
                    new_boxes.append(boxes[index])
                    new_scores.append(scores[index])
                    break
                
                else:
                    width1 = boxes[index][3]-boxes[index][1]

                    res1 = abs(self.stats['width_mean']-width1)

                    width2 = boxes[index2][3]-boxes[index2][1]

                    res2 = abs(self.stats['width_mean']-width2)

                    if res1>res2:
                        skip_index.append(index2)
                    else:
                        break
        
        
        if len(boxes)-1 not in skip_index:
            new_boxes.append(boxes[-1])
            new_scores.append(scores[-1])

        self.results['detection_boxes'] = new_boxes
        self.results['detection_scores'] = new_scores
        return self.results          

    def remove_boundary(self):
        """remove either left most or right most or both boxes, depending on the distance between digits

        Returns:
            results: removed boxes of the boundary
        """

        xmin = [arr[1] for arr in self.results['detection_boxes']]
        xmax = [arr[3] for arr in self.results['detection_boxes']]

        # print(xmax,xmin)

        xdis = [max(xmin[i+1] - xmax[i],0) for i in range(len(xmax)-1)]
        temp_xdis = xdis[1:-1]
        xdis_mean = np.mean(temp_xdis)
        xdis_std = np.std(temp_xdis)

        # print(xdis)
        # print(xdis_mean, xdis_std)

        new_boxes = self.results['detection_boxes']
        new_scores = self.results['detection_scores']

        if xdis[0]>=xdis_mean+2.5*xdis_std:
            new_boxes.pop(0)
            new_scores.pop(0)
        if xdis[-1]>=xdis_mean+2.5*xdis_std:
            new_boxes.pop(-1)
            new_scores.pop(-1)
        
        self.results = {
            'detection_boxes': new_boxes,
            'detection_scores': new_scores
        }

        return self.results