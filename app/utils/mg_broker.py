import pika
import urllib.parse 
from app.core.constants import MG_BROKER_HOST, MG_BROKER_USER, MG_BROKER_PASS, MG_BROKER_PORT, MG_BROKER_LOG_DB_QUEUE


def get_rabbitmq_connection():
    password = urllib.parse.quote(MG_BROKER_PASS)
    amq_url = f"amqp://{MG_BROKER_USER}:{password}@{MG_BROKER_HOST}:{MG_BROKER_PORT}/"
    parameters = pika.URLParameters(amq_url)
    connection = pika.BlockingConnection(parameters)
    return connection


# def create_rabbitmq_channel():
#     connection = get_rabbitmq_connection()
#     channel = connection.channel()
#     exchange_name = "fastapi_exchange"
#     exchange_type = "direct"
#     channel.exchange_declare(exchange_name, exchange_type, durable=True)
#     channel.queue_declare(MG_BROKER_LOG_DB_QUEUE, durable=True)
#     channel.queue_bind(MG_BROKER_LOG_DB_QUEUE, exchange_name, routing_key=MG_BROKER_LOG_DB_QUEUE)
#     return channel, exchange_name

def create_rabbitmq_channel():
    connection = get_rabbitmq_connection()
    channel = connection.channel()
    return channel, "fastapi_exchange"


def send_message_to_rabbitmq(message):
    channel, exchange_name = create_rabbitmq_channel()
    channel.basic_publish(
        exchange_name,
        routing_key=MG_BROKER_LOG_DB_QUEUE,
        body=message.encode("utf-8"),
        # properties=pika.BasicProperties(delivery_mode=2),
    )
    channel.close()
