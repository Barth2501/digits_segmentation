import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def segment(img, save=False,name=None):
        """
        This method is responsible for licence plate segmentation with histogram of pixel projection approach
        :param img: input image
        :return: list of image, each one contain a digit
        """
        # list that will contains all digits
        caracter_list_image = list()

        # img = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=BLACK)
        if len(img.shape)==3:
            # change to gray
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Change to numpy array format
            nb = np.array(gray)
            # get height and weight
            x = gray.shape[1]
            y = gray.shape[0]
        else:
            # Change to numpy array format
            nb = np.array(img)
            # get height and weight
            x = img.shape[1]
            y = img.shape[0]

        # Check if the background is in black or white

        total = np.mean(nb)
        # Binarization
        if total > 255/2:
            # this number are experimental and seems to work well with white background
            nb[nb >= np.quantile(nb, 0.46)] = 255
            nb[nb < np.quantile(nb, 0.46)] = 0
            nb = cv2.bitwise_not(nb)
        else:
            # this number are experimental and seems to work well with black background
            nb[nb > np.quantile(nb, 0.78)] = 255
            nb[nb < np.quantile(nb, 0.78)] = 0

        # compute the sommation
        y_sum = cv2.reduce(nb, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

        y_sum = y_sum / x

        # convert y_arr to numpy array
        w = np.array(y_sum)

        # convert to zero small details and 1 for needed details
        w[w < 30] = 0
        w[w > 30] = 1

        # Find the zone of interest in the image
        t2 = list()
        f = 0
        ff = w[0]
        for i in range(w.size):
            if w[i] != ff:
                f += 1
                ff = w[i]
                t2.append(i)
        rect_v = np.array(t2)

        # take the appropriate height
        rectv = []
        rectv.append(rect_v[0])
        rectv.append(rect_v[1])
        max = int(rect_v[1]) - int(rect_v[0])
        for i in range(len(rect_v) - 1):
            diff2 = int(rect_v[i + 1]) - int(rect_v[i])

            if diff2 > max:
                rectv[0] = rect_v[i]
                rectv[1] = rect_v[i + 1]
                max = diff2

        # crop the image
        nb = nb[rectv[0]-2:rectv[1]+2,:]
        w = w[rectv[0]-2:rectv[1]+2]

        x_sum = cv2.reduce(nb, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

        # rotate the vector x_sum
        x_sum = x_sum.transpose()

        # division the result by height and weight
        x_sum = x_sum / y

        # convert x_sum to numpy array
        z = np.array(x_sum)

        # convert to zero small details
        z[z < np.quantile(z,0.3)] = 0
        z[z > np.quantile(z,0.3)] = 1

        # vertical segmentation
        test = z.transpose() * nb

        # horizontal segmentation
        test = w * test

        # Check if the character detection have been done correctly
        cv2.imshow('Binarization of the license plate',test)
        cv2.waitKey(2000)

        # Character detection along the horizontal axis
        f = 0
        ff = z[0]
        t1 = list()
        for i in range(z.size):
            if z[i] != ff:
                f += 1
                ff = z[i]
                t1.append(i)
        rect_h = np.array(t1)

        # extract caracter
        for i in range(len(rect_h) - 1):

            # eliminate slice that can't be a digit, a digit must have width bigger then 8
            diff1 = int(rect_h[i + 1]) - int(rect_h[i])

            if (diff1 > 2) and (z[rect_h[i]] == 1):
                # cutting nb (image) and adding each slice to the list caracter_list_image
                caracter_list_image.append(nb[:, rect_h[i]-1:rect_h[i + 1]+1])

                # draw rectangle on digits
                cv2.rectangle(img, (rect_h[i], rectv[0]), (rect_h[i + 1], rectv[1]), (0, 255, 0), 1)

        # Show segmentation result
        # image = plt.imshow(img)
        # plt.show(image)

        return caracter_list_image

def crop_char_and_save(character, name=None):

    BLACK = [0, 0, 0]
    character = cv2.copyMakeBorder(character, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=BLACK)

    x, y = character.shape
    y_sum = cv2.reduce(character, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    w = y_sum / x

    # convert to zero small details and 1 for needed details
    w[w < 5] = 0
    w[w > 5] = 1

    # Find the zone of interest in the image
    t2 = list()
    f = 0
    ff = w[0]
    for i in range(w.size):
        if w[i] != ff:
            f += 1
            ff = w[i]
            t2.append(i)
    rect_v = np.array(t2)

    # take the appropriate height
    rectv = []
    rectv.append(rect_v[0])
    if len(rect_v)<2:
        return None
        
    rectv.append(rect_v[1])
    maxi = int(rect_v[1]) - int(rect_v[0])
    for i in range(len(rect_v) - 1):
        diff2 = int(rect_v[i + 1]) - int(rect_v[i])
        if diff2 > maxi:
            rectv[0] = rect_v[i]
            rectv[1] = rect_v[i + 1]
            maxi = diff2

    new_char = character[rectv[0]:rectv[1],:]

    # We add a black border to make the image closer to a square

    img = cv2.copyMakeBorder(new_char, max(0,(y-x)//2)+1, max(0,(y-x)//2)+1, max(0,(x-y)//2)+1, max(0,(x-y)//2)+1, cv2.BORDER_CONSTANT, value=BLACK)
    if name:
        print('image saved in ' + name + '.jpg')
        cv2.imwrite(name + '.jpg', img)

    return img

def MinMaxGray(img):
    maxi = -np.inf
    mini = np.inf
    for i, row in enumerate(img):
        for j, cell in enumerate(row):
            if cell > maxi:
                maxi = cell
            elif cell < mini and cell != 0:
                mini = cell
    return mini, maxi
    
def expansion_img(img):
    exp_img = img.copy()
    mini, maxi = MinMaxGray(img)
    aug_factor = 255/(maxi-mini)
    for i, row in enumerate(img):
        for j, cell in enumerate(row):
            exp_img[i,j] = (cell - mini) * aug_factor
    return img

def egalisation_img(img):

    # Calculate its cumulative histogram
    histogram, edges = np.histogram(img, bins=256)
    histogram = histogram/sum(histogram)
    c_hist = np.cumsum(histogram)
    
    # Compute the egalisation
    eg_img = img.copy()
    for i, row in enumerate(img):
        for j, cell in enumerate(row):
            eg_img[i,j] = int(255*c_hist[cell])
    return eg_img

def rotate_img(img):

    # Expand the image
    exp_img = expansion_img(img)
    
    # Egalize the image
    eg_img = egalisation_img(exp_img)

    binar_img = img.copy()

    # I am converting the image into three differents scale of gray
    first_thres = np.quantile(eg_img,0.33)
    second_thres = np.quantile(eg_img,0.67)
    
    binar_img[eg_img>second_thres] = 255
    binar_img[(eg_img>first_thres) & (eg_img<=second_thres)] = 130
    binar_img[eg_img<=first_thres] = 0

    # Calculating L(x,y+1)-L(x,y-1) and L(x+1,y)-L(x-1,y)
    Ix = cv2.Sobel(binar_img, cv2.CV_64F, 2, 0, ksize=5)
    Iy = cv2.Sobel(binar_img, cv2.CV_64F, 0, 2, ksize=5)

    # Calculating the rotation theta of every point
    theta = np.arctan(Iy/Ix)
    theta = np.nan_to_num(theta)

    # Converting from radian to degrees
    theta = theta*180/np.pi

    # Rotating the image according to the mean rotation angle
    # of the image
    h,w = img.shape[:2]
    center = (w/2,h/2)
    M = cv2.getRotationMatrix2D(center, -theta.mean(), 1)
    rotated_img = cv2.warpAffine(img, M, (w,h))
    return rotated_img

def black_or_white(img):
    """
    This function is used to know if the digits on the license plate are written 
    in black or in white. If they are in white we will inverse the grays of the 
    image
    """
    res_img = img.copy()

    # We vertically sum the gray value of the image and retrieve 
    # only the middle part of the image
    sum_x_img = list(np.sum(img, axis=1).astype(np.int16))
    middle_sum_x_img = sum_x_img[len(sum_x_img)*1//4:len(sum_x_img)*3//4+1]
    
    # We wmoothen the curve
    yhat = savgol_filter(middle_sum_x_img, 3, 1)

    # And calculate its derivative
    d_yhat = [(yhat[i+2]-yhat[i])/2 for i in range(len(yhat)-2)]

    # We calculate the sum of half the image in order to know
    # if the global sign is positive or negative
    first_middle_sum = sum(d_yhat[:len(d_yhat)//2+1])
    sec_middle_sum = sum(d_yhat[len(d_yhat)//2:])

    # This means the digits are black
    if first_middle_sum<0 and sec_middle_sum>0:
        res_img = 255 - img.copy()
    # This means the digits are white
    elif first_middle_sum>0 and sec_middle_sum<0:
        pass
    else:
        raise ValueError('We cannot know if the digits of the license plate are black or white, see black_of_white function in the documentation')
    return res_img
        

def cropping_border(img):
   
    # This list contains the gray color accumulated along the rows
    # A low number will mean a dark row and a high number will mean 
    # a light row 
    sum_x_img = list(np.sum(img, axis=1).astype(np.int16))

    # Check where the function reaches its first minimum
    i = 0
    while sum_x_img[i+1]>sum_x_img[i]:
        i += 1
    while sum_x_img[i+1]<sum_x_img[i]:
        i += 1
    # Check where the function reaches its last minimum
    j = len(sum_x_img)-1
    while sum_x_img[j-1]>sum_x_img[j]:
        j -= 1
    while sum_x_img[j-1]<sum_x_img[j]:
        j -= 1
    # Return the image cropped
    return img[i:j+1,:]

def discrete_to_continu(y,x):
    # make a discrete function, from a list, continuous
    #Check if the requested point is in the curve 
    if x>=len(y) or x<0:
        return None
    if isinstance(x,int):
        return y[x]
    # We extrapolate the point if the requested number is not int
    else:
        if int(x)+1<len(y):
            return (y[int(x)+1]-y[int(x)])*x + (int(x)+1)*y[int(x)] - int(x)*y[int(x)+1]
        else:
            return None

def gradient_descent(y,start,eps,pas):
    # implementation od the gradient descent to find a local minimum
    x1 = start
    if discrete_to_continu(y,x1+1) and discrete_to_continu(y,x1-1):
        dy = (discrete_to_continu(y,x1+1)-discrete_to_continu(y,x1-1))/2
        x2 = x1 - pas*dy
    else:
        return None
    if discrete_to_continu(y,x2) and discrete_to_continu(y,x1):
        diff = abs(discrete_to_continu(y,x2)-discrete_to_continu(y,x1))
    else:
        return None
    while diff>eps:
        x1 = x2
        if discrete_to_continu(y,x2+1) and discrete_to_continu(y,x2-1):
            dy = (discrete_to_continu(y,x2+1)-discrete_to_continu(y,x2-1))/2
            x2 = x1 - pas*dy
        else:
            return None
        if discrete_to_continu(y,x2) and discrete_to_continu(y,x1):
            diff = abs(discrete_to_continu(y,x2)-discrete_to_continu(y,x1))
        else:
            return None
    return x2,discrete_to_continu(y,x2)

def best_global_min(y,eps,pas):
    # We repeat the gradient descent on mutliple point
    index, best_min = None,np.inf
    for i in range(0,len(y),5):
        if gradient_descent(y,i,eps,pas) and gradient_descent(y,i,eps,pas)[1] < best_min:
            index,best_min = gradient_descent(y,i,eps,pas)
    return index

def get_last_local_min(y,eps,pas):
    """
    This function is used to find the last local min of a function
    it uses multiple starting points and uses gradient descent to find local mins
    it returns the last it finds
    """ 
    index,value = 0,0
    for i in range(0,len(y),5):
        if gradient_descent(y,i,eps,pas):
            index,value = gradient_descent(y,i,eps,pas)
    return index,value

def find_threshold(img):
    """
    This function is used to find the best threshold for the binarization
    it calculates the global minimum of the derivative of the cumulative histogram
    We find this minimum using gradient descent
    """
    # Calculate the cumulative histogram 
    histogram, edges = np.histogram(img, bins=80)
    histogram = histogram/sum(histogram)
    c_hist = np.cumsum(histogram)
    # Compute its derivative function to find global minimum
    deriv_c_hist = [(c_hist[i]-c_hist[i-4])/4 for i in range(4,len(c_hist))]
    # Smoothen the derivative curve
    yhat = savgol_filter(deriv_c_hist, 15, 3)
    # Find the last local global min of the function 
    threshold = (get_last_local_min(yhat,1e-6,1000)[0]-2)*img.max()/len(yhat) 
    return threshold

def binarization(img):
    # Transform the image into a binarized image
    bin_img = img.copy()
    threshold = find_threshold(img)
    bin_img[img>threshold] = 255
    bin_img[img<=threshold] = 0
    return bin_img

def longuest_sub_array(arr):
    """
    This function caculates the longuest sub array in a list

    A sub array is an array which doesn't contain any 0

    It returns the index of the last element which composes the sub array
    and the length of the sub array
    """
    sub_arrays = []
    last_index = 0
    length = 0
    for i,l in enumerate(arr):
        if l != 0 :
            length += 1
            last_index = i
        else:
            if last_index == i-1:
                sub_arrays.append((last_index, length))
            length=0
    if sub_arrays == [(0,0)]:
        print('The image cannot be cropped vertically')
        return None
    return max(sub_arrays, key=lambda p: p[1])

def vertical_crop(img):
    """
    This function crops vertically the input image

    It is based on the longuest sub array which is present in the image
    """
    sum_x_img = np.sum(img, axis=1)
    if longuest_sub_array(sum_x_img):
        last_index, length = longuest_sub_array(sum_x_img)
        first_index = last_index - length + 1
        return img[first_index:last_index+1,:]
    else:
        return img

def horizontal_crop(img,l):
    """
    This function crops horizontally the input image

    It is based on the cumulative sum of the rows, if its slope is too low,
    that means there is no digits in this area
    """
    L=[]
    sum_y_img = np.sum(img,axis=0)
    csum_y_img = sum_y_img.cumsum()
    mean_slope = (csum_y_img[-1]-csum_y_img[0])/len(csum_y_img)*0.33
    for i in range(len(csum_y_img)-l):
        slope = (csum_y_img[i+l]-csum_y_img[i])/l
        if slope < mean_slope:
            L.append(i)
    for i in range(1,len(L)):
        if L[i]!=L[i-1]+1:
            begin = i-1
            break
    for i in range(len(L)-1,1,-1):
        if L[i]!=L[i-1]+1:
            last = i
            break
    return img[:,L[begin]+l+1:L[last]+1]

import itertools

def dfs(pos,img,list_of_all):
    """
    This function performs the the deep first search of the white
    neighbours of a pixel
    """
    h_max,w_max,h_min,w_min = pos[0],pos[1],pos[0],pos[1]
    for i,j in itertools.product([-1,0,1],[-1,0,1]):
        if ((i,j) != (0,0) 
                and (pos[0]+i,pos[1]+j) in list_of_all
                and pos[0]+i>=0
                and pos[0]+i<img.shape[0]
                and pos[1]+j>=0
                and pos[1]+j<img.shape[1]
                and img[pos[0]+i,pos[1]+j] == 255):
            list_of_all.remove((pos[0]+i,pos[1]+j))
            new_h_max,new_w_max,new_h_min,new_w_min,list_of_all = dfs((pos[0]+i,pos[1]+j),img,list_of_all)
            h_max = max(h_max,new_h_max)
            w_max = max(w_max,new_w_max)
            h_min = min(h_min,new_h_min)
            w_min = min(w_min,new_w_min)
    return h_max,w_max,h_min,w_min,list_of_all
    
def segmentation(img):
    """
    This function will explore every pixels of the image
    to build trees. As soon as a tree is fully explored, its dimension are
    put inside a list
    """
    list_of_all = [(i,j) for i in range(img.shape[0]) for j in range(img.shape[1])]
    digits_list = []
    while list_of_all:
        next_pos = list_of_all.pop(0)
        new_h_max,new_w_max,new_h_min,new_w_min,list_of_all = dfs(next_pos,img,list_of_all)
        if new_h_max != new_h_min and new_w_max!=new_w_min:
            digits_list.append(img[new_h_min:new_h_max+1,new_w_min:new_w_max+1])
    return digits_list

def remove_noises(list_of_cropped):
    """
    This function cleans the list of cropped output by removing from it 
    the output which doesn't contain more than eight pixels
    """

    cleaned_list = []
    for caracter in list_of_cropped:
        if np.sum(caracter)>255*8:
            cleaned_list.append(caracter)
    return cleaned_list

