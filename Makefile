#!make

crawl:
	python3 data/crawler.py ${nb_pages}
.PHONY: crawl
