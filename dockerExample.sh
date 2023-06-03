
# remove existing container
docker rmi --force cspot
docker system prune --all --force --volumes

# build a new image
docker build -t nirmallab/cspot:20230329 -t nirmallab/cspot:latest .

# login to push
docker login

# push to docker hub
docker push nirmallab/cspot --all-tags

# pull from docker hub
docker pull nirmallab/cspot:latest


# interactive
docker run -it --mount type=bind,source=/Users/aj/Desktop/csExampleData,target=/data --name cspot cspot /bin/bash

# full script
export projectDir="/Users/aj/Desktop/csExampleData"
docker run -it --mount type=bind,source=$projectDir,target=/$projectDir \
                cspot \
                python /app/generateThumbnails.py \
                --spatialTablePath $projectDir/quantification/exampleSpatialTable.csv \
                --imagePath $projectDir/image/exampleImage.tif \
                --markerChannelMapPath $projectDir/markers.csv \
                --markers ECAD CD3D \
                --maxThumbnails 100 \
                --projectDir $projectDir





