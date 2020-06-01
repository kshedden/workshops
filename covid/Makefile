xetex=/nfs/kshedden/texlive/2019/bin/x86_64-linux/xelatex

covid_growth:
	jupytext --execute --to notebook python/covid_growth.py
	mv python/covid_growth.ipynb notebooks
	jupyter-nbconvert --to latex notebooks/covid_growth.ipynb
	cd notebooks; $(xetex) covid_growth.tex;  $(xetex) covid_growth.tex; cd ..

testing_mortality_usa:
	jupytext --execute --to notebook python/testing_mortality_usa.py
	mv python/testing_mortality_usa.ipynb notebooks
	jupyter-nbconvert --to latex notebooks/testing_mortality_usa.ipynb
	cd notebooks; $(xetex) testing_mortality_usa.tex;  $(xetex) testing_mortality_usa.tex; cd ..

testing_mortality_simulation:
	jupytext --execute --to notebook python/testing_mortality_simulation.py
	mv python/testing_mortality_simulation.ipynb notebooks
	jupyter-nbconvert --to latex notebooks/testing_mortality_simulation.ipynb
	cd notebooks; $(xetex) testing_mortality_simulation.tex;  $(xetex) testing_mortality_simulation.tex; cd ..

testing_mortality_world:
	jupytext --execute --to notebook python/testing_mortality_world.py
	mv python/testing_mortality_world.ipynb notebooks
	jupyter-nbconvert --to latex notebooks/testing_mortality_world.ipynb
	cd notebooks; $(xetex) testing_mortality_world.tex;  $(xetex) testing_mortality_world.tex; cd ..
