#!/usr/local/bin/python3
import numpy as np
import cv2
import click

@click.group()
def main():
	"""Application entry point."""
	pass

def save_and_show(filename, img, show):
	"""Save image to disk or show it onscreen if parameter is true."""
	if show:
		cv2.imshow('Output', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	else:
		cv2.imwrite(
			'{0}_processed.{1}'.format(
				filename[:filename.rfind('.')],
				filename[filename.rfind('.') + 1:]
				),
			img)

# YIQ<->RGB conversion matrix values
mYIQ = np.array([
    0.299,  0.587,  0.114,
    0.596, -0.274, -0.322,
    0.211, -0.523,  0.312
	]).reshape((3,3))

mRGB = np.array([
	1,  0.956,  0.0621,
	1, -0.272, -0.6470,
	1, -1.106,  1.7030
	]).reshape((3,3))

@click.command()
@click.argument('file')
@click.option('--show', is_flag=True, default=False)
def RGBtoYIQ(file, show):
	"""(1.1) to YIQ."""
	img = RGBtoYIQinternal(cv2.imread(file))
	save_and_show(file, img, show)

def RGBtoYIQinternal(img):
	w = img.shape[0]
	h = img.shape[1]
	aux = []

	for pixel in img.reshape((w * h, 3)):
		result = np.dot(mYIQ, pixel / 255)
		aux.append(result)

	imgYIQ = np.array(aux).reshape((w, h, 3)).astype('float64')
	return imgYIQ

@click.command()
@click.argument('file')
@click.option('--show', is_flag=True, default=False)
def YIQtoRGB(file, show):
	"""(1.1) to RGB"""
	img = YIQtoRGBinternal(cv2.imread(file))
	save_and_show(file, img, show)

def YIQtoRGBinternal(img):
	w = img.shape[0]
	h = img.shape[1]
	aux = []

	for pixel in img.reshape((w * h, 3)):
		new = np.dot(mRGB, pixel) * 255

		for j in range(len(new)):
			if new[j] > 255:
				new[j] = 255
			elif new[j] < 0:
				new[j] = 0

		aux.append(new)

	imgRGB = np.array(aux).reshape((w,h,3)).astype('int32')
	return imgRGB

@click.command()
@click.argument('file')
@click.option('--show', is_flag=True, default=False)
@click.option('-r', is_flag=True, default=False)
@click.option('-g', is_flag=True, default=False)
@click.option('-b', is_flag=True, default=False)
@click.option('-grayscale', is_flag=True, default=False)
def monochromatic(file, show, r, g, b, grayscale):
	"""(1.2) select a channel of color"""
	img =  cv2.imread(file)
	w = img.shape[0]
	h = img.shape[1]
	img = img.reshape((h*w,3))
	for pixel in range(h*w):
		img[pixel][0] = img[pixel][0] if b else 0
		img[pixel][1] = img[pixel][1] if g else 0
		img[pixel][2] = img[pixel][2] if r else 0
	img = img.reshape((w,h,3))

	if grayscale:
		img = img.reshape((h*w,3))
		for pixel in range(w*h):
			rgb_sum = (img[pixel][0] + img[pixel][1] + img[pixel][2])

			img[pixel][0] = rgb_sum
			img[pixel][1] = rgb_sum
			img[pixel][2] = rgb_sum
		img = img.reshape((w,h,3))
	
	save_and_show(file, img, show)

@click.command()
@click.argument('file')
@click.option('--show', is_flag=True, default=False)
@click.option('-yiq', is_flag=True, default=False)
def invert(file, show, yiq):
	"""(1.3) Invert colors."""
	img = cv2.imread(file)

	if yiq:
		img = RGBtoYIQinternal(img)
		channels = 1
		max_value = 1
	else:
		channels = 3
		max_value = 255

	width = img.shape[0]
	height = img.shape[1]

	for x in range(1, width-1):
		for y in range(1, height-1):
			for channel in range (0, channels):
				original = img.item(x, y, channel)
				new = max_value - original
				img.itemset(x, y, channel, new)

	if yiq:
		img = YIQtoRGBinternal(img)

	save_and_show(file, img, show)

@click.command()
@click.argument('file')
@click.argument('c')
@click.option('--show', is_flag=True, default=False)
@click.option('-yiq', is_flag=True, default=False)
def brightness_add(file, c, show, yiq):
	"""(1.4) Brightness control by adding."""
	try:
		c = int(c)
	except ValueError:
		print('ERROR: c must be an integer.')
		return

	img = cv2.imread(file)

	if yiq:
		img = RGBtoYIQinternal(img)
		channels = 1
	else:
		channels = 3

	width = img.shape[0]
	height = img.shape[1]

	for x in range(1, width-1):
		for y in range(1, height-1):
			for channel in range (0, channels):
				original = img.item(x, y, channel)
				new = original + c

				if new > 255:
					new = 255

				img.itemset(x, y, channel, new)

	if yiq:
		img = YIQtoRGBinternal(img)

	save_and_show(file, img, show)

@click.command()
@click.argument('file')
@click.argument('c')
@click.option('--show', is_flag=True, default=False)
@click.option('-yiq', is_flag=True, default=False)
def brightness_mult(file, c, show, yiq):
	"""(1.4) Brightness control by multiplying."""
	try:
		c = float(c)

		if c < 0:
			raise ValueError
	except ValueError:
		print('ERROR: c must be a positive float.')
		return

	img = cv2.imread(file)

	if yiq:
		img = RGBtoYIQinternal(img)
		channels = 1
	else:
		channels = 3

	width = img.shape[0]
	height = img.shape[1]

	for x in range(1, width-1):
		for y in range(1, height-1):
			for channel in range (0, channels):
				original = img.item(x, y, channel)
				new = original * c

				if new > 255:
					new = 255

				img.itemset(x, y, channel, new)

	if yiq:
		img = YIQtoRGBinternal(img)

	save_and_show(file, img, show)

@click.command()
@click.argument('file')
@click.option('-m', default=None)
@click.option('--show', is_flag=True, default=False)
def threshold(file, m, show):
	"""(1.4) Threshold."""
	img = cv2.imread(file)
	img = RGBtoYIQinternal(img)
	width = img.shape[0]
	height = img.shape[1]

	if not m:
		y_sum = 0
		for x in range(1, width-1):
			for y in range(1, height-1):
				y_sum += img.item(x, y, 0)

		m = y_sum / (width * height)

	try:
		m = int(m)

		if m < 0:
			raise ValueError
	except ValueError:
		print('ERROR: m must be a positive integer.')
		return

	for x in range(1, width-1):
		for y in range(1, height-1):
			original = img.item(x, y, 0)
			if original > m:
				new = original
			else:
				new = 0

			img.itemset(x, y, 0, new)

	img = YIQtoRGBinternal(img)
	save_and_show(file, img, show)

def convolution(m,ksize,operation):
	"""The main operation for filtering."""
	step = 1 #there's a sort of problem to solve to implements others steps
	h = m.shape[0]
	w = m.shape[1]
	x = ksize[0]
	y = ksize[1]
	border_x = x // 2
	border_y = y // 2
	print(w,h,x,y,border_x,border_y)
	m0 = np.pad(m[:,:,0],pad_width=border_x,mode="edge")
	m1 = np.pad(m[:,:,1],pad_width=border_x,mode="edge")
	m2 = np.pad(m[:,:,2],pad_width=border_x,mode="edge")
	kernel0 = np.zeros(x*y).reshape(ksize)
	kernel1 = np.zeros(x*y).reshape(ksize)
	kernel2 = np.zeros(x*y).reshape(ksize)
	for i in range(h):
		for j in range(w):
			kernel0 = m0[i : i + y, j: j + x]
			kernel1 = m1[i : i + y, j: j + x]
			kernel2 = m2[i : i + y, j: j + x]
			m[i,j,0] = operation(kernel0)
			m[i,j,1] = operation(kernel1)
			m[i,j,2] = operation(kernel2)
	return m

def mean(matrix):
	return matrix.mean()

def median(matrix):
	return np.median(matrix)

def sobel(matrix):
	sobel = np.array([1,0,-1,2,0,-2,1,0,-1])
	matrix = matrix.reshape(matrix.shape[0] * matrix.shape[1]).astype('float64')
	matrix = matrix ** sobel
	r = matrix.astype('int32').sum()
	return 255 - r if r < 256 else 0

def laplacian(matrix):
	laplac = np.array([0,-1,0,-1,4,-1,0,-1,0]).reshape((3,3))
	matrix = matrix.astype('float64')
	matrix = matrix ** laplac
	r = matrix.astype('int32').sum()
	return 255 - r if r < 256 else 0

@click.command()
@click.argument('file')
@click.argument('operation', default='mean')
@click.argument('size', default=3)
@click.option('--show', is_flag=True, default=False)
def filter(file,operation,size,show):
	"""(1.5 and 1.6) filters"""
	print(operation)
	img = cv2.imread(file)
	size = (size,size)
	if operation == 'mean':
		img = convolution(img,size,mean)
	elif operation == 'median':
		img = convolution(img,size,median)
	elif operation == 'sobel':
		if size != (3,3):
			print('WARNING: Sobel filter only works with a 3x3 kernel size.')
			size = (3,3)

		img = convolution(img,size,sobel)
	elif operation == 'lapla':
		if size != (3,3):
			print('WARNING: Laplacian filter only works with a 3x3 kernel size.')
			size = (3,3)

		img = convolution(img,size,laplacian)

	save_and_show(file,img,show)

# Add command line options
main.add_command(invert)
main.add_command(brightness_add)
main.add_command(brightness_mult)
main.add_command(threshold)
main.add_command(RGBtoYIQ)
main.add_command(YIQtoRGB)
main.add_command(monochromatic)
main.add_command(filter)

if __name__ == '__main__':
	main()
