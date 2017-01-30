#!/usr/bin/env python
# Resize all passed in images to 64x64 grayscale pngs named '####.png'
# monotonically increasing in the specified output folder
from optparse import OptionParser
import PIL
import os
from PIL import Image

usage = "usage: %prog [options] image1 [image2 ...]"
parser = OptionParser(usage=usage)
parser.add_option('-o', '--output_folder', dest='output_folder',
    help='Output folder to save resized images to', default='')
parser.add_option('-n', '--max_number', type="int", dest='max_n',
    help='Maximum number of images to process', default=-1)
parser.add_option('-d', '--dryrun', action="store_true", dest='dryrun',
    help='Do a dry run (no processing/saving)', default=False)


def main():
  (options, args) = parser.parse_args()
  if options.dryrun:
    print('Doing dryrun')

  n = len(args)
  if n ==0:
    parser.print_help()
    return
  if options.max_n >= 0:
    print('Processing %d of %d files...' % (options.max_n, n))
  else:
    print('Processing %d files...' % n)

  i = 0;
  for filename in args:
    try:
      print('%04d' % i)
      print('\tOpening %s' % filename)
      # Load image
      img = PIL.Image.open(filename)
      if img.mode == 'L':
        raise Exception('\tFlicker image ignored')
      # Resize to 64x64
      img_out = img.resize((64,64),PIL.Image.BILINEAR)
      # Convert to rgb if not
      # if img_out.mode != 'RGB':
      #   img_out = img_out.convert('RGB')
      # Convert to grayscale
      img_out = img_out.convert('L')

      # Save to output
      out_path = os.path.join(options.output_folder, '%04d.png' % i)
      if not options.dryrun:
        img_out.save(out_path)
        print('\tSaved to %s' % out_path)
      i = i + 1
      if (options.max_n >=0 and i >= options.max_n):
        break
    except Exception as e:
      print(e)
      print('\tSkipping %s' % filename)

  print("Done processing %d files." % i)

if __name__ == '__main__':
    main()