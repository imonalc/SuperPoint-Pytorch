import numpy as np
import cv2

def get_theta(x, y):
    theta = np.where(y < 0, (-1) * np.arctan2(y, x), 2 * np.pi - np.arctan2(y, x))
    return theta

def create_equirectangular_to_bottom_and_top_map(input_w, input_h, output_sqr, z):
    x, y = np.meshgrid(np.linspace(-output_sqr/2.0, output_sqr/2.0-1, output_sqr), 
                       np.linspace(-output_sqr/2.0, output_sqr/2.0-1, output_sqr), indexing='ij')
    z = np.full(x.shape, z)
    
    rho = np.sqrt(x**2 + y**2 + z**2)
    norm_theta = get_theta(x, y) / (2 * np.pi)
    norm_phi = (np.pi - np.arccos(z / rho)) / np.pi
    ix = norm_theta * input_w
    iy = norm_phi * input_h

    ix = np.where(ix >= input_w, ix - input_w, ix)
    iy = np.where(iy >= input_h, iy - input_h, iy)
    
    return ix, iy

def create_equirectangular_to_front_and_back_map(input_w, input_h, output_sqr, x):
    z, y = np.meshgrid(np.linspace(-output_sqr/2.0, output_sqr/2.0-1, output_sqr), 
                       np.linspace(-output_sqr/2.0, output_sqr/2.0-1, output_sqr), indexing='ij')
    x = np.full(z.shape, x)
    
    rho = np.sqrt(x**2 + y**2 + z**2)
    norm_theta = get_theta(x, y) / (2 * np.pi)
    norm_phi = (np.pi - np.arccos(z / rho)) / np.pi
    ix = norm_theta * input_w
    iy = norm_phi * input_h

    ix = np.where(ix >= input_w, ix - input_w, ix)
    iy = np.where(iy >= input_h, iy - input_h, iy)
    
    return ix, iy

def create_equirectangular_to_left_and_right_map(input_w, input_h, output_sqr, y):
    x, z = np.meshgrid(np.linspace(-output_sqr/2.0, output_sqr/2.0-1, output_sqr), 
                       np.linspace(-output_sqr/2.0, output_sqr/2.0-1, output_sqr), indexing='ij')
    y = np.full(x.shape, y)
    
    rho = np.sqrt(x**2 + y**2 + z**2)
    norm_theta = get_theta(x, y) / (2 * np.pi)
    norm_phi = (np.pi - np.arccos(z / rho)) / np.pi
    ix = norm_theta * input_w
    iy = norm_phi * input_h

    ix = np.where(ix >= input_w, ix - input_w, ix)
    iy = np.where(iy >= input_h, iy - input_h, iy)
    
    return ix, iy


def create_cube_map(back_img, bottom_img, front_img, left_img, right_img, top_img, output_sqr):
    cube_map_img = np.zeros((3 * output_sqr, 4 * output_sqr, 3))
    cube_map_img[output_sqr:2*output_sqr, 3*output_sqr:4*output_sqr] = back_img
    cube_map_img[2*output_sqr:3*output_sqr, output_sqr:2*output_sqr] = bottom_img
    cube_map_img[output_sqr:2*output_sqr, output_sqr:2*output_sqr] = front_img
    cube_map_img[output_sqr:2*output_sqr, 0:output_sqr] = left_img
    cube_map_img[output_sqr:2*output_sqr, 2*output_sqr:3*output_sqr] = right_img
    cube_map_img[0:output_sqr, output_sqr:2*output_sqr] = top_img
    return cube_map_img


def create_cube_imgs(img):
    if len(img.shape) == 2:
        input_h, input_w = img.shape
    elif len(img.shape) == 3:
        input_h, input_w, _ = img.shape
    output_sqr = int(input_w / 4)
    normalized_f = 1

    z = (output_sqr / (2.0 * normalized_f))
    bottom_map_x, bottom_map_y = create_equirectangular_to_bottom_and_top_map(input_w, input_h, output_sqr, z)
    bottom_img = cv2.remap(img, bottom_map_x.astype("float32"), bottom_map_y.astype("float32"), cv2.INTER_CUBIC)

    z = (-1) * (output_sqr / (2.0 * normalized_f))
    top_map_x, top_map_y = create_equirectangular_to_bottom_and_top_map(input_w, input_h, output_sqr, z)
    top_img = cv2.remap(img, top_map_x.astype("float32"), top_map_y.astype("float32"), cv2.INTER_CUBIC)
    top_img = cv2.flip(top_img, 0)

    x = (-1) * (output_sqr / (2.0 * normalized_f))
    front_map_x, front_map_y = create_equirectangular_to_front_and_back_map(input_w, input_h, output_sqr, x)
    front_img = cv2.remap(img, front_map_x.astype("float32"), front_map_y.astype("float32"), cv2.INTER_CUBIC)

    x = output_sqr / (2.0 * normalized_f)
    back_map_x, back_map_y = create_equirectangular_to_front_and_back_map(input_w, input_h, output_sqr, x)
    back_img = cv2.remap(img, back_map_x.astype("float32"), back_map_y.astype("float32"), cv2.INTER_CUBIC)
    back_img = cv2.flip(back_img, 1)

    y = (-1) * (output_sqr / (2.0 * normalized_f))
    left_map_x, left_map_y = create_equirectangular_to_left_and_right_map(input_w, input_h, output_sqr, y)
    left_img = cv2.remap(img, left_map_x.astype("float32"), left_map_y.astype("float32"), cv2.INTER_CUBIC)
    left_img = cv2.rotate(left_img, cv2.ROTATE_90_CLOCKWISE)

    y = output_sqr / (2.0 * normalized_f)
    right_map_x, right_map_y = create_equirectangular_to_left_and_right_map(input_w, input_h, output_sqr, y)
    right_img = cv2.remap(img, right_map_x.astype("float32"), right_map_y.astype("float32"), cv2.INTER_CUBIC)
    right_img = cv2.flip(right_img, 1)
    right_img = cv2.rotate(right_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return [front_img, left_img, back_img, right_img, top_img, bottom_img], output_sqr


def create_3dmap_from_size_np(img_w, img_h):
    h = np.linspace(-np.pi/2, np.pi/2, img_h)
    w = np.linspace(-np.pi, np.pi, img_w)
    
    h += (np.pi/2) / img_h
    w += np.pi / img_w
    
    theta, phi = np.meshgrid(w, h, indexing="ij")
    
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)
    
    return x, y, z

def padding_cube_np(img):
    if img.ndim == 3:
        h, w, c = img.shape
    elif img.ndim == 2:
        h, w = img.shape
        c = 1
        img = img[:, :, np.newaxis]
    cw = w // 4
     
    canvas = np.zeros((h+4, w+4, c), dtype=img.dtype)
    canvas[2:-2, 2:-2,:] = img
     
    # up    
    canvas[0:2,cw+2:2*cw+2,:] = np.rot90(img[cw:cw+2, 3*cw:,:], 2)
    # bottom
    canvas[-2:,cw+2:2*cw+2,:] = np.rot90(img[2*cw-2:2*cw,3*cw:,:], 2)
    # left
    canvas[cw+2:2*cw+2,0:2,:] = img[cw:2*cw,-2:,:]
    # right
    canvas[cw+2:2*cw+2,-2:,:] = img[cw:2*cw,0:2,:]
 
    canvas[cw:cw+2,:cw+2,:] = np.rot90(canvas[:cw+2,cw+2:cw+4,:])
    canvas[:cw+2,cw:cw+2,:] = np.rot90(canvas[cw+2:cw+4,:cw+2,:],3)
    
    canvas[2*cw+2:2*cw+4,:cw+2,:] = np.rot90(canvas[2*cw+2:,cw+2:cw+4,:],3)
    canvas[2*cw+2:,cw:cw+2,:] = np.rot90(canvas[2*cw:2*cw+2,:cw+2,:])
    
    canvas[cw:cw+2,2*cw+2:3*cw+2,:] = np.rot90(canvas[2:cw+2,2*cw:2*cw+2,:],3)
    canvas[:cw+2,2*cw+2:2*cw+4:] = np.rot90(canvas[cw+2:cw+4,2*cw+2:3*cw+4,:])
    
    canvas[2*cw+2:2*cw+4,2*cw+2:3*cw+2,:] = np.rot90(canvas[2*cw+2:-2,2*cw:2*cw+2,:])
    canvas[2*cw+2:,2*cw+2:2*cw+4,:] = np.rot90(canvas[2*cw:2*cw+2,2*cw+2:3*cw+4,:], 3)
 
    canvas[cw:cw+2, 3*cw+2:,:] = canvas[3:1:-1, 2*cw+1:cw-1:-1,:]
    canvas[2*cw+2:2*cw+4, 3*cw+2:,:] = canvas[-3:-5:-1, 2*cw+1:cw-1:-1,:]

    if c == 1:
        canvas = canvas.reshape((h+4, w+4))
     
    return canvas

def cube_to_equirectangular_np(img, width):
    img_array = np.array(img).astype(np.float32)/ 255.0
    img_w = width
    img_h = width // 2
    width = img_array.shape[1] // 4

    x, y, z = create_3dmap_from_size_np(img_w, img_h)
    w = 0.5

    # front
    xx = w*y / x + w
    yy = w*z / x + w    
    mask = (xx > 0) & (xx < 1) & (yy > 0) & (yy < 1) & (x > 0)
    tmpx = np.where(mask, xx*width + width, 0)
    tmpy = np.where(mask, yy*width + width, 0)
     
    # back
    xx = w*y / x + w
    yy = -w*z / x + w    
    mask = (xx > 0) & (xx < 1) & (yy > 0) & (yy < 1) & (x < 0)
    tmpx = np.where(mask, xx*width + width*3, tmpx)
    tmpy = np.where(mask, yy*width + width, tmpy)
     
    # right
    xx = -w*x / y + w
    yy = w*z / y + w    
    mask = (xx > 0) & (xx < 1) & (yy > 0) & (yy < 1) & (y > 0)
    tmpx = np.where(mask, xx*width + width*2, tmpx)
    tmpy = np.where(mask, yy*width + width, tmpy)
     
    # left
    xx = -w*x / y + w
    yy = -w*z / y + w    
    mask = (xx > 0) & (xx < 1) & (yy > 0) & (yy < 1) & (y < 0)
    tmpx = np.where(mask, xx*width, tmpx)
    tmpy = np.where(mask, yy*width + width, tmpy)
     
    # up
    xx = -w*y / z + w
    yy = -w*x / z + w    
    mask = (xx > 0) & (xx < 1) & (yy > 0) & (yy < 1) & (z < 0)
    tmpx = np.where(mask, xx*width + width, tmpx)
    tmpy = np.where(mask, yy*width, tmpy)
     
    # bottom
    xx = w*y / z + w
    yy = -w*x / z + w    
    mask = (xx > 0) & (xx < 1) & (yy > 0) & (yy < 1) & (z > 0)
    tmpx = np.where(mask, xx*width + width, tmpx)
    tmpy = np.where(mask, yy*width + width*2, tmpy)

    tmpx += 2.0 - 0.5
    tmpy += 2.0 - 0.5
    tmpx = tmpx.astype(np.float32)
    tmpy = tmpy.astype(np.float32)

    cube = padding_cube_np(img_array)

    ret_img = cv2.remap(cube, tmpx, tmpy, interpolation=cv2.INTER_LINEAR)
    ret_img = cv2.rotate(ret_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ret_img = cv2.flip(ret_img, 0)
    
    return ret_img