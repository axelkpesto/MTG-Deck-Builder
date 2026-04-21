FROM php:8.3-apache

RUN a2enmod rewrite headers

RUN apt-get update && apt-get install -y --no-install-recommends \
        libcurl4-openssl-dev \
    && docker-php-ext-install curl \
    && rm -rf /var/lib/apt/lists/*

COPY config.php /var/www/config.php
COPY public /var/www/html

RUN printf '#!/bin/bash\n\
set -e\n\
PORT=${PORT:-8080}\n\
sed -i "s/Listen 80/Listen ${PORT}/" /etc/apache2/ports.conf\n\
sed -i "s/<VirtualHost \*:80>/<VirtualHost *:${PORT}>/" /etc/apache2/sites-available/000-default.conf\n\
exec apache2-foreground\n' > /start.sh \
    && chmod +x /start.sh

CMD ["/start.sh"]
