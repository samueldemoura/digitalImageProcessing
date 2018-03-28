#!/usr/local/bin/python3
import numpy as np
import cv2
import click

@click.group()
def main():
	"""Application entry point."""
	pass

def save_and_show(filename, img, save, show):
	"""Save image to disk and show it if parameters are true."""
	if save:
		cv2.imwrite(
			'{0}_processed.{1}'.format(
				filename[:filename.rfind('.')],
				filename[filename.rfind('.') + 1:]
				),
			img)

	if show:
		cv2.imshow('Output', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

@click.command()
@click.argument('file')
@click.option('--save/--dont-save', default=True)
@click.option('--show', is_flag=True, default=False)
def invert(file, save, show):
	"""(1.3) Invert colors."""
	img = cv2.imread(file)

	width = img.shape[0]
	height = img.shape[1]

	for x in range(1, width-1):
	    for y in range(1, height-1):
	        for channel in range (0, 3):
	            original = img.item(x, y, channel)
	            new = 255 - original
	            img.itemset(x, y, channel, new)

	save_and_show(file, img, save, show)

@click.command()
@click.argument('file')
@click.argument('c')
@click.option('--save/--dont-save', default=True)
@click.option('--show', is_flag=True, default=False)
def brightness_add(file, c, save, show):
	"""(1.4) Brightness control by adding."""
	try:
		c = int(c)
	except ValueError:
		print('ERROR: c must be an integer.')
		return

	img = cv2.imread(file)
	width = img.shape[0]
	height = img.shape[1]

	for x in range(1, width-1):
	    for y in range(1, height-1):
	        for channel in range (0, 3):
	            original = img.item(x, y, channel)
	            new = original + c

	            if new > 255:
	            	new = 255

	            img.itemset(x, y, channel, new)

	save_and_show(file, img, save, show)

@click.command()
@click.argument('file')
@click.argument('c')
@click.option('--save/--dont-save', default=True)
@click.option('--show', is_flag=True, default=False)
def brightness_mult(file, c, save, show):
	"""(1.4) Brightness control by multiplying."""
	try:
		c = float(c)

		if c < 0:
			raise ValueError
	except ValueError:
		print('ERROR: c must be a positive float.')
		return

	img = cv2.imread(file)
	width = img.shape[0]
	height = img.shape[1]

	for x in range(1, width-1):
	    for y in range(1, height-1):
	        for channel in range (0, 3):
	            original = img.item(x, y, channel)
	            new = original * c

	            if new > 255:
	            	new = 255

	            img.itemset(x, y, channel, new)

	save_and_show(file, img, save, show)

@click.command()
@click.argument('file')
@click.option('-m', default='None')
@click.option('--save/--dont-save', default=True)
@click.option('--show', is_flag=True, default=False)
def threshold(file, m, save, show):
	"""(1.4) Threshold."""
	if m == None:
		print('Must calculate m')
		return

	try:
		m = int(m)

		if m < 0:
			raise ValueError
	except ValueError:
		print('ERROR: m must be a positive integer.')
		return

	img = cv2.imread(file)
	width = img.shape[0]
	height = img.shape[1]

	for x in range(1, width-1):
	    for y in range(1, height-1):
	        for channel in range (0, 3):
	            original = img.item(x, y, channel)
	            if original > m:
	            	new = 255
	            else:
	            	new = 0

	            img.itemset(x, y, channel, new)

	save_and_show(file, img, save, show)

# Add command line options
main.add_command(invert)
main.add_command(brightness_add)
main.add_command(brightness_mult)
main.add_command(threshold)

if __name__ == '__main__':
	main()