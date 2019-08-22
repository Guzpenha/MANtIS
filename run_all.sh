#!/usr/bin/env bash

bash fetch_stackexchange_dump.sh apple \
&& python run.py json apple \
&& bash fetch_stackexchange_dump.sh dba \
&& python run.py json dba \
&& bash fetch_stackexchange_dump.sh diy \
&& python run.py json diy \
&& bash fetch_stackexchange_dump.sh electronics \
&& python run.py json electronics \
&& bash fetch_stackexchange_dump.sh english \
&& python run.py json english \
&& bash fetch_stackexchange_dump.sh gaming \
&& python run.py json gaming \
&& bash fetch_stackexchange_dump.sh gis \
&& python run.py json gis \
&& bash fetch_stackexchange_dump.sh math \
&& python run.py json math \
&& bash fetch_stackexchange_dump.sh physics \
&& python run.py json physics \
&& bash fetch_stackexchange_dump.sh scifi \
&& python run.py json scifi \
&& bash fetch_stackexchange_dump.sh security \
&& python run.py json security \
&& bash fetch_stackexchange_dump.sh stats \
&& python run.py json stats \
&& bash fetch_stackexchange_dump.sh travel \
&& python run.py json travel \
&& bash fetch_stackexchange_dump.sh workplace \
&& python run.py json workplace \
&& bash fetch_stackexchange_dump.sh worldbuilding \
&& python run.py json worldbuilding