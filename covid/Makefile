xetex=/nfs/kshedden/texlive/2019/bin/x86_64-linux/xelatex

covid_growth:
	jupytext --execute --to notebook python/covid_growth.py
	mv python/covid_growth.ipynb notebooks
	jupyter-nbconvert --to latex notebooks/covid_growth.ipynb
	cd notebooks; $(xetex) covid_growth.tex;  $(xetex) covid_growth.tex; cd ..

mort_testing:
	jupytext --execute --to notebook python/mort_testing.py
	mv python/mort_testing.ipynb notebooks
	jupyter-nbconvert --to latex notebooks/mort_testing.ipynb
	cd notebooks; $(xetex) mort_testing.tex;  $(xetex) mort_testing.tex; cd ..
