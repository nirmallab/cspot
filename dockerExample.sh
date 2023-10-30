
# remove existing container
docker rmi --force cspot
docker system prune --all --force --volumes

# build a new image
docker build -t nirmallab/cspot:20230601 -t nirmallab/cspot:latest .

# login to push
docker login

# push to docker hub
docker push nirmallab/cspot --all-tags

# pull from docker hub
docker pull nirmallab/cspot:latest

# interactive
docker run -it --mount type=bind,source=C:/Users/ajit/Desktop/cspotExampleData,target=/data --name cspot nirmallab/cspot /bin/bash
docker run -it --mount type=bind,source=/Users/aj/Documents/cspotExampleData,target=/data --name cspot nirmallab/cspot /bin/bash

docker run -it --mount type=bind,source=/Users/aj/Documents/cspotExampleData,target=/data --name cspot cspot /bin/bash


# full script
# specify the directory where the sample data lives and Run the docker command
export projectDir="C:/Users/ajit/Desktop/cspotExampleData"



export projectDir="C:/Users/ajit/Desktop/cspotExampleData"

export projectDir="/mnt/c/Users/aj/Desktop/cspotExampleData"

export projectDir="/Users/aj/Desktop/cspotExampleData"

export projectDir="C:/Users/ajit/Desktop/cspotExampleData"
docker run -it --mount type=bind,source=$projectDir,target=/$projectDir \
                nirmallab/cspot:latest \
                python /app/generateThumbnails.py \
                --spatialTablePath $projectDir/quantification/exampleSpatialTable.csv \
                --imagePath $projectDir/image/exampleImage.tif \
                --markerChannelMapPath $projectDir/markers.csv \
                --markers ECAD CD3D \
                --maxThumbnails 100 \
                --projectDir $projectDir


# command promt
set projectDir=C:/Users/ajit/Desktop/cspotExampleData
docker run -it --mount type=bind,source=%projectDir%,target=/%projectDir% nirmallab/cspot:latest python /app/generateThumbnails.py --spatialTablePath %projectDir%/quantification/exampleSpatialTable.csv --imagePath %projectDir%/image/exampleImage.tif --markerChannelMapPath %projectDir%/markers.csv --markers ECAD CD3D --maxThumbnails 100 --projectDir %projectDir%


# poweshell
$projectDir = "C:\Users\aj\Desktop\cspotExampleData"
docker run -it --mount type=bind,source=$projectDir,target=/$projectDir nirmallab/cspot:latest python /app/generateThumbnails.py --spatialTablePath $projectDir\quantification\exampleSpatialTable.csv --imagePath $projectDir\image\exampleImage.tif --markerChannelMapPath $projectDir\markers.csv --markers ECAD CD3D --maxThumbnails 100 --projectDir $projectDir

# step-2 onwards
export projectDir="/Users/aj/Desktop/cspotExampleData"

set projectDir=C:/Users/ajit/Desktop/cspotExampleData
docker run -it --mount type=bind,source=%projectDir%,target=/%rojectDir% nirmallab/cspot:latest python /app/generateTrainTestSplit.py --thumbnailFolder %projectDir%/CSPOT/Thumbnails/CD3D %projectDir%/CSPOT/Thumbnails/ECAD --projectDir %projectDir%


set projectDir=C:/Users/ajit/Desktop/cspotExampleData
docker run -it --mount type=bind,source=$projectDir,target=/$projectDir \
                nirmallab/cspot:latest \
                python /app/csTrain.py \
                --trainingDataPath $projectDir/CSPOT/TrainingData \
                --projectDir $projectDir \
                --epochs=1