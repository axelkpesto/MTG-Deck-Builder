#!/bin/bash
set -euo pipefail

PORT=${PORT:-8080}
sed -i "s/Listen 80/Listen ${PORT}/" /etc/apache2/ports.conf
sed -i "s/<VirtualHost \*:80>/<VirtualHost *:${PORT}>/" /etc/apache2/sites-available/000-default.conf

if [ -n "${GCS_DATA_BUCKET:-}" ]; then
    echo "Downloading card image dataset from gs://${GCS_DATA_BUCKET}"
    mkdir -p /var/www/data

    TOKEN=$(curl -sf \
        -H "Metadata-Flavor: Google" \
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
        | php -r "echo json_decode(file_get_contents('php://stdin'))->access_token;")

    curl -sf \
        -H "Authorization: Bearer ${TOKEN}" \
        "https://storage.googleapis.com/storage/v1/b/${GCS_DATA_BUCKET}/o/data%2Fprocessed%2Fcard_images.json?alt=media" \
        -o /var/www/data/card_images.json

    echo "Card image dataset ready."
fi

exec apache2-foreground
