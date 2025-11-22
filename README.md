# recovery_scripts
These Python scripts apply methods of statistical analysis to digital images to recover damaged content. I developed them working with ChatGPT and on a Mac using Python's IDLE Shell. They 
have been written to work with 8- and 16-bit tiff images.

To run the scripts, make sure that the necessary python libraries are installed. Normally, I install them through Mac's Terminal (press COMMAND+spacebar, type Terminal in the Spotlight Search and hit return.

Make sure your python is up-to-date. You can type:
python3 -m pip install --upgrade pip

For the LDA script, you can install the needed libraries in one command:
python3 -m pip install numpy matplotlib scikit-learn opencv-python

For the Alt Color Space script (splits RGB tiff into the three channels for four different alternative color spaces: Opponent Color Space, CMYK, HSV, LAB, LUV, and YCrCb):
python3 -m pip install numpy tiffile tkinter

I've generated short videos for running the PCA, MNF, MAF, and ICA script:  ; and the LDA script: https://youtu.be/r3n9DJVhhH4.
Furthermore, I have a short video on how to use ImageJ to combine results into pseudocolor images to enhance visibility of damaged content: https://youtu.be/2-QL849HePM?si=ARxA6aruGHbw0lt_.
