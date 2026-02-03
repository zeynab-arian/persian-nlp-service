from sqlalchemy.orm import Session
from sqlalchemy import text
from core.services import epic_service
from db.session import get_db_masterdb
from fastapi import Request

def get_web_user_id_by_person(db: Session, person_id: int) -> int | None:
    sql = text("SELECT WebUserID FROM framework.AccountSpecs WHERE PersonID = :pid LIMIT 1")
    result = db.execute(sql, {"pid": person_id}).scalar()
    return result

def send_message_to_user(
    request: Request,
    message: str,
    creator_user_id: str,
    ref_id: int,
    message_link: str,
    message_group_code: int,
    web_user_id: int,
) -> int:
    db = next(get_db_masterdb(request))
    try:
        insert_sql = text("""
            INSERT INTO framework.messages 
            (Message, CreatorUserID, RefID, MessageLink, LinkTarget, MessageGroupCode)
            VALUES (:message, :creator_user_id, :ref_id, :message_link, '_blank', :message_group_code)
        """)
        db.execute(insert_sql, {
            "message": message,
            "creator_user_id": creator_user_id,
            "ref_id": ref_id,
            "message_link": message_link,
            "message_group_code": message_group_code,
        })
        db.commit()

        # گرفتن آخرین ID درج شده
        msg_id = db.execute(text("SELECT LAST_INSERT_ID()")).scalar()

        insert_user_sql = text("""
            INSERT INTO framework.UserMessages (MessageID, UserID)
            VALUES (:msg_id, :user_id)
        """)
        db.execute(insert_user_sql, {"msg_id": msg_id, "user_id": web_user_id})
        db.commit()

        return msg_id
    finally:
            db.close()