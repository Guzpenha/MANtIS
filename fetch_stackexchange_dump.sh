#!/usr/bin/env bash

topic=$1

allowed_topics=( \
    "academia" "android" "apple" "askubuntu" "bicycles" "biology" "buddhism" "cooking" "dba" "diy" "electronics" "ell" "economics" \
    "english" "gaming" "gis" "math" "security" "law" "money" "movies" "music" "philosophy" \
    "physics" "photo" "politics" "salesforce" "scifi" "security" "sound" "stats" "travel" "workplace" "worldbuilding")

checkTopic() {
    local e match="$1"
    shift
    for e; do [[ "$e" == "$match" ]] && return 0; done
    echo "== Topic not allowed. Choose 1 from the following list =="
    echo "== ${allowed_topics[@]} =="
    exit 1
}

topic=$1

checkTopic "$topic" "${allowed_topics[@]}"

install_util="k"
#read -p "This script requires the p7zip utility to unzip the dump: Do you wish to install it? (y\n): " install_util

if [[ "$install_util" == "y" ]]; then
    case "$OSTYPE" in
    "darwin"*)
        echo "Mac system detected. Using homebrew."
        brew update && brew install p7zip
        ;;
    "linux-gnu")
        echo "Linux system detected. Using apt. Your admin password will be asked in order to install."
        sudo apt-get install p7zip-full
        ;;
    esac
fi

DIRECTORY=./stackexchange_dump/

if [[ ! -d "$DIRECTORY" ]]; then
    mkdir ./stackexchange_dump
fi

cd ./stackexchange_dump/

if [[ ! -d "$topic" ]]; then
    mkdir "$topic"
fi

cd "$topic"

echo "Initiating the download of the stackExchange dump"

# Fetch and extract dataset
if [[ "$topic" == "askubuntu" ]]; then
    curl -L0 "https://archive.org/download/stackexchange/$topic.com.7z" -o dump.7z       && \
    7z x dump.7z && \
    rm dump.7z && \
    rm Badges.xml PostLinks.xml PostHistory.xml Tags.xml
else
    curl -L0 "https://archive.org/download/stackexchange/$topic.stackexchange.com.7z" -o dump.7z       && \
    7z x dump.7z && \
    rm dump.7z && \
    rm Badges.xml PostLinks.xml PostHistory.xml Tags.xml
fi

curl -L0 https://archive.org/download/stackexchange/readme.txt -o schema.txt




