# FCC_flowers
Final project for 1-credit python course.

# DOCUMENTATION
My ultimate objective for this code (which still failed) was to read in large iage datasets of both daisies and dandelions, have the user input an image of the flower they wish to identify, and have the computer recognize if the plant is LESS likely to be a weed (arbitrary probability of below 75%) and should not be sprayed with weedkiller.

I played with using docker to install and operate tensorflow locally, and got PRETTY DARN CLOSE (messy thought processes and other files included in "valiant efforts."

In the end, I used Jupyter (which I learned how to use with terminal through this FCC! Very handy!) to grab tensorflow and keras and build my own Convolutional Neural Net rather than using one of tensorflow's (at first I was hoping to use Inception).

I fought with the SETUP for a lot of this for much longer than 20 hours (all somewhat last-minute too, admittedly; patly my own doing and partly a brutual dance schedule), and pretty significant syntax issues relating to the TRAINING STEP and PREDICTION() have persisted as of 2am the morning the FCC was due...

NEVERTHELESS, this entire experience has been extremely educational for me - first, to understand that my proposal was overwhelmingly ambitious because of the sheer fact that I didn't understand what a neural network even was, much less how much effort it takes for amateur coders just to differentiate between two types of flowers, when google does this every millisecond with millions upon  millions of images (maybe billions?). 

Not to mention, neural networks can be applied to other formats than just images. This is a bit too abstract for me, and I appreciate knowing that my skill in python extends far enough that I can process logic easily and execute mathematical, graphical, and data-related funcitonalities. I will leave the computer architecture and machine learning to specialists (who hopefully I will collaborate with in the future!). In which case, my more refined understanding of tensorflow NOW will benefit me in the long run, no matter how frustrating and stressful just some SYNTAX ended up being.

# How to run the code (such as it is)
You may need to install/upgrade
1. Tensorflow
2. Keras
3. image (pillow)

*I believe if you just download the script CNN_DaisyVsWeed.py and run, you will be able to see how far I have gotten. I have also included the .ipynb file, if you have jupyter installed.*

Feel free to peruse the 'valiant effort,' and I apologize for not being able to be proactive enough to work around like-avoidable issues with Setup (unless my 12-inch MacBook is truly not cut out for this stuff, though I doubt that's the issue).

Thanks for a great semester; I must stress despite my progress on this project that I have learned a lot about python, and expect for this to be a great advantage for me when it comes to doing environmental research in the future!

-- Mariana Silva
