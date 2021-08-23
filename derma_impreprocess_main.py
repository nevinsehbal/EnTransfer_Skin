import argparse
import derma_impreprocess_functions as functions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_folder_path', help='path to folder including images')
    opt = parser.parse_args()
    imdir = opt.images_folder_path
    imageslist = functions.readDirectory(imdir)
    #print(imageslist)
    functions.imagesConvertSquare(imageslist,imdir,save=True)
    #functions.imagesRotate()
    #functions.imagesFlip(0,1)
    #functions.imagesBlur()
    #functions.images()
