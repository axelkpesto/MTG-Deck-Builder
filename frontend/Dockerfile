FROM php:8.3-apache

RUN a2enmod rewrite headers

RUN apt-get update && apt-get install -y --no-install-recommends \
        libcurl4-openssl-dev \
    && docker-php-ext-install curl \
    && rm -rf /var/lib/apt/lists/*

COPY config.php /var/www/config.php
COPY public /var/www/html
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
