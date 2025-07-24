import requests

def get_socialverse_posts():
    url = "https://api.socialverseapp.com/posts/summary/get?page=1&page_size=1000"
    headers = {
        "Flic-Token": "flic_b1c6b09d98e2d4884f61b9b3131dbb27a6af84788e4a25db067a22008ea9cce5"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()  
    else:
        return {"error": response.status_code, "message": response.text}

