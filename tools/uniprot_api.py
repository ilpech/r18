import requests, sys
import json

def getGene(uniprot_id):
    params = {
        'gene': uniprot_id,
        'chromosome': 18
    }
    requestURL = "https://www.ebi.ac.uk/proteins/api/coordinates"
    r = requests.get(
        requestURL, 
        headers={"Accept" : "application/json"},
        params=params
    )
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    responseBody = r.json()
    out = '../data/api_out/{}.json'.format(uniprot_id)
    with open(out, 'w') as f:
        json.dump(responseBody, f)
    print(responseBody)
    print('written out', out)
    
if __name__ == '__main__':
    getGene('Q8WU67')