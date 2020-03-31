# Vehicle_detection
Entry->[trainSVMFile->extractFeatures[pos+neg]->trainSVM->saveModel]->detector[loadSavedModel->generateSlidingWindows->extractFeatures->standardize->predict->drawBoxes Based On Prediction With Sliding Windows As Reference]->save the coords-> ID generation and tracking.
