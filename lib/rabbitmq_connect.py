import pika
from retrying import retry


class RabbitServerConnector:

    @staticmethod
    @retry(stop_max_delay=120000)
    def get_connection(rabbit_host):
        """
        retrying max 2 minutes, to prevent RabbitMQ service is not ready while connecting
        :return: connection to rabbitmq server
        """
        return pika.BlockingConnection(pika.ConnectionParameters(rabbit_host))
