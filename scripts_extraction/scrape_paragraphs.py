import csv
import requests
from bs4 import BeautifulSoup
import re
import time
from tqdm import tqdm


initial_paragraph = "https://digitalzibaldone.net/node/p2700_1"

output_paragraphs = []
places = []
persons = []
works = []

def scrape_paragraphs_recursive(paragraph_num):
    time.sleep(5)
    if paragraph_num != None:
        response = requests.get(paragraph_num)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            node = soup.find('div', id=paragraph_num.replace("https://digitalzibaldone.net/node/", ""))
            if node:
                text = re.sub("\s+", " ", node.text)
                text = re.sub(r"^\[.*?\]\s", "", text)
                output_paragraphs.append({"doc_id": paragraph_num, "text": text, "publication_date":1823})
                links = node.find_all("a")
                for link in links:
                    href = link.get("href")
                    if href.startswith("Q"):
                        surface = re.sub("\s+", " ", link.text.strip())
                        places.append({"surface": surface, "_id": href, "par_id": paragraph_num})
                    elif "person" in link.get("class", []):
                        surface = re.sub("\s+", " ", link.text.strip())
                        persons.append({"surface": surface, "_id": href, "par_id": paragraph_num})
                    elif href.startswith("/node/") and not href.startswith("/node/p"):
                        surface = re.sub("\s+", " ", link.text.strip())
                        works.append({"surface": surface, "_id": href.replace("/node/", ""), "par_id": paragraph_num})
            try:
                nextprev_div = soup.find('div', class_='nextprev')
                second_link = nextprev_div.find_all('a')[1]  # Secondo link
                second_link_href = second_link.get("href")
                page_num = re.match("https:\/\/digitalzibaldone\.net\/node\/p(.*?)\_.*?", second_link_href).group(1)
                if int(page_num)>3000:
                    next_page = None
                else:
                    next_page = second_link_href
                print(next_page)
                scrape_paragraphs_recursive(next_page)
            except:
                next_page = None
                scrape_paragraphs_recursive(next_page)
    else:
        print("error")
        return None

scrape_paragraphs_recursive(initial_paragraph)

if len(output_paragraphs)>0:
    o_keys = output_paragraphs[0].keys()
    with open("./DZ/paragraphs_test.csv", "w", encoding="utf-8") as f:
        dict_writer = csv.DictWriter(f, o_keys)
        dict_writer.writeheader()
        dict_writer.writerows(output_paragraphs)

pbar = tqdm(total=len(output_paragraphs))
annotations = list()
for par in output_paragraphs:
    annotated_chars = set()
    _id = par["doc_id"]
    text = par["text"]
    # match labels of places in paragraph
    for loc in places:
        if loc["par_id"]==_id:
            pattern = loc["surface"]
            for match in re.finditer(pattern+"\W", text):
                surface = match.group(0)[:-1]
                start = match.start()
                end = match.end()-1
                char_ranges = set((range(start, end+1)))
                # make sure no annotation overlaps
                if len(annotated_chars.intersection(char_ranges)) == 0:
                    annotations.append(
                        {"doc_id":_id, "surface": surface, "start_pos": start, "end_pos": end, "identifier": loc["id"],
                         "type": "LOC"})
                    annotated_chars.update(char_ranges)
    for per in persons:
        if per["par_id"]==_id:
            pattern = per["surface"]
            for match in re.finditer(pattern+"\W", text):
                surface = match.group(0)[:-1]
                start = match.start()
                end = match.end()-1
                char_ranges = set((range(start, end+1)))
                # make sure no annotation overlaps
                if len(annotated_chars.intersection(char_ranges)) == 0:
                    annotations.append(
                        {"doc_id":_id, "surface": surface, "start_pos": start, "end_pos": end, "identifier": per["id"],
                         "type": "PER"})
                    annotated_chars.update(char_ranges)
    for work in works:
        if work["par_id"]==_id:
            pattern = work["surface"]
            for match in re.finditer(pattern+"\W", text):
                surface = match.group(0)[:-1]
                start = match.start()
                end = match.end()-1
                char_ranges = set((range(start, end+1)))
                if len(annotated_chars.intersection(char_ranges)) == 0:
                    annotations.append(
                        {"doc_id":_id, "surface": surface, "start_pos": start, "end_pos": end, "identifier": work["id"],
                         "type": "WORK"})
                    annotated_chars.update(char_ranges)
    pbar.update(1)
pbar.close()


if len(annotations)>0:
    keys = annotations[0].keys()
    with open("./DZ/annotations_test.csv", "w", encoding="utf-8") as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(annotations)
    f.close()