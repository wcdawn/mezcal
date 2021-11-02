NOTE = conduction_validation

# PdfLaTeX compilation options
LATEX = pdflatex
LATEXOPT = -halt-on-error -file-line-error

PYTHON = python3

FIGS_DIR = ../figs
PYTHON_DIR = ../python

FIGURES = $(FIGS_DIR)/radial_line.pdf \
		  $(FIGS_DIR)/radial_circle.pdf \
		  $(FIGS_DIR)/linear.pdf \
		  $(FIGS_DIR)/var_cond_temperature.pdf \
		  $(FIGS_DIR)/var_cond_conductivity.pdf \
		  $(FIGS_DIR)/var_cond_diff.pdf

EXTRAS = 

AUXILIARY = $(NOTE).log $(NOTE).aux

#=================================================================
# Generate PDF of manuscript using PdfLaTeX
#=================================================================

default : all

all : pdf

pdf : $(NOTE).pdf

$(NOTE).pdf : $(NOTE).tex $(FIGURES) $(EXTRAS)
	$(LATEX) $(LATEXOPT) -draftmode $(NOTE)
	@#bibtex $(NOTE)
	@#makeglossaries $(NOTE)
	$(LATEX) $(LATEXOPT) -draftmode $(NOTE)
	$(LATEX) $(LATEXOPT) $(NOTE)
	@grep -i --color=always 'warn' $(NOTE).log || true
	@grep -i --color=always 'overfull' $(NOTE).log || true
	@grep -i --color=always 'underfull' $(NOTE).log || true

#=================================================================
# Other
#=================================================================

$(FIGS_DIR)/radial_line.pdf : $(PYTHON_DIR)/conduction_cyl.py
	$(PYTHON) $<

$(FIGS_DIR)/radial_circle.pdf : $(PYTHON_DIR)/conduction_cyl.py
	$(PYTHON) $<

$(FIGS_DIR)/linear.pdf : $(PYTHON_DIR)/conduction.py
	$(PYTHON) $<

$(FIGS_DIR)/linear_geometry.pdf : $(PYTHON_DIR)/conduction.py
	$(PYTHON) $<

$(FIGS_DIR)/var_cond_temperature.pdf : $(PYTHON_DIR)/var_cond.py
	$(PYTHON) $<

$(FIGS_DIR)/var_cond_conductivity.pdf : $(PYTHON_DIR)/var_cond.py
	$(PYTHON) $<

$(FIGS_DIR)/var_cond_diff.pdf : $(PYTHON_DIR)/var_cond.py
	$(PYTHON) $<

clean :
	$(RM) $(AUXILIARY)
	$(RM) $(FIGURES)

cleanpdf :
	$(MAKE) clean
	$(RM) $(NOTE).pdf

spellcheck :
	aspell -c $(manuscript).tex --mode=tex

.PHONY: all clean
