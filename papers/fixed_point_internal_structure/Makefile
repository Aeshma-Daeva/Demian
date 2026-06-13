PDF=main.pdf

.PHONY: all figures clean

all: figures $(PDF)

figures:
	../../venv/bin/python generate_figures.py

$(PDF): main.tex references.bib figures
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
