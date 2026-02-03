from fastapi import APIRouter
from app.core.services.domain_manager import save_domain, list_domains, delete_domain, add_keyword, list_domains_with_keywords
from app.core.services.domain_search import search_text_in_domain

router = APIRouter(tags=["Extraction"])

@router.post("/search")
def search_in_domain(domain_name: str, text: str):
    results, text_keywords, detailed_matches, matched_keywords = search_text_in_domain(domain_name, text)
    return {
        "domain": domain_name,
        "results": results,
        "text_keywords": text_keywords,
        "detailed_matches": detailed_matches,
        "matched_keywords": matched_keywords  
    }

@router.post("/add_domain")
def add_domain(domain_name: str, keywords: list[str]):
    save_domain(domain_name, keywords)
    return {"status": "success", "domain": domain_name}

@router.post("/add_keyword")
def add_domain_keyword(domain_name: str, keyword: str):
    add_keyword(domain_name, keyword)
    return {"status": "success", "domain": domain_name, "keyword": keyword}

@router.get("/list_domains")
def get_domains():
    return {"domains": list_domains_with_keywords()}

@router.delete("/delete_domain/{domain_name}")
def remove_domain(domain_name: str):
    delete_domain(domain_name)
    return {"status": "success", "domain": domain_name}
