# Запуск ноды на Linux

Обычная нода Parano1d проверяет полные блоки, поддерживает Live State,
ретранслирует транзакции и обслуживает синхронизацию. Она не майнит.

Это руководство устанавливает официальный релиз Core как системную службу на
64-битном сервере Linux с systemd.

## Требования

Штатная реализация системы доказательств требует:

- x86-64 с SSE4.1 и PCLMULQDQ; либо
- ARM64 с NEON и PMULL.

При запуске автоматически выбирается `pclmul`, `avx2+vpclmul`,
`avx512bw+vpclmul` или `neon+pmull`. Скалярная эталонная реализация в
штатной ноде не используется.

P2P-соединения принимаются на TCP `9600`. JSON-RPC должен оставаться на
`127.0.0.1:9601`.

Основной изменяемый объём хранилища определяется размером Live State.
Компактные заголовки по 212 байт хранятся постоянно, а полные тела блоков —
только для последних 18 блоков.

До заказа виртуальной машины и выбора лимитов прочитайте
[Оборудование и ресурсы](hardware.md).

## Установка релиза Core

Скачайте архив нужной архитектуры и `SHA256SUMS` со
[страницы релизов](https://github.com/ignotusnemo/elide/releases). Проверьте
архив до распаковки, заменив `VERSION` номером версии:

```sh
grep '  elide-core-vVERSION-linux-x86_64.tar.gz$' SHA256SUMS \
  | sha256sum --check
```

Команда должна вывести `OK`. Для ARM64 замените `linux-x86_64` на
`linux-aarch64`.

Распакуйте архив и проверьте оборудование:

```sh
tar -xzf elide-core-vVERSION-linux-x86_64.tar.gz
./elide --check-hardware
```

На поддерживаемой машине отчёт заканчивается:

```text
NODE READY
```

Установите ноду и CLI:

```sh
sudo install -m 0755 elide elide-cli /usr/local/bin/
```

## Системный пользователь

Храните данные ноды отдельно от интерактивных пользователей:

```sh
sudo useradd --system --home-dir /var/lib/elide \
  --create-home --shell /usr/sbin/nologin elide
sudo install -d -o elide -g elide -m 0700 /var/lib/elide
sudo install -d -o root -g elide -m 0750 /etc/elide
```

Создайте `/etc/elide/elide.toml`:

```toml
[network]
listen = "0.0.0.0:9600"
seeds = []

[storage]
backend = "mdbx"
path = "/var/lib/elide"

[rpc]
listen = "127.0.0.1:9601"

[mining]
enabled = false
miner_address = ""
```

Защитите конфигурацию:

```sh
sudo chown root:elide /etc/elide/elide.toml
sudo chmod 0640 /etc/elide/elide.toml
```

Указывать сид вручную не нужно. Релизный бинарник находит публичную сеть через
встроенные DNS-сиды и запоминает успешные исходящие пиры.

## Запуск под systemd

Создайте `/etc/systemd/system/elide.service`:

```ini
[Unit]
Description=Parano1d node
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=elide
Group=elide
ExecStart=/usr/local/bin/elide --config /etc/elide/elide.toml
Restart=on-failure
RestartSec=5
KillSignal=SIGINT
TimeoutStopSec=45
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

`KillSignal=SIGINT` оставляет ноде время корректно закрыть сетевые службы и
сбросить MDBX.

Загрузите файл службы и запустите ноду:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now elide
sudo systemctl status elide
```

Следите за запуском и синхронизацией:

```sh
sudo journalctl -u elide -f
```

## Проверка ноды

По умолчанию CLI подключается к локальному RPC:

```sh
elide-cli status
elide-cli peers
elide-cli state
```

`status` должен показывать актуальную высоту, число в `peers` должно стать
ненулевым, а `state` сообщает аутентифицированные размеры Live State.

## Сетевой доступ

Разрешите входящий TCP `9600` в межсетевых экранах сервера и провайдера. За
NAT перенаправьте TCP `9600` на ноду. Синхронизация работает и при наличии только
исходящих соединений, но входящие пиры делают ноду полезной для сети.

Не публикуйте TCP `9601`. Для удалённого администрирования используйте
SSH-туннель или другой аутентифицированный приватный транспорт.

## Остановка и обновление

Перед заменой бинарников или копированием данных остановите службу:

```sh
sudo systemctl stop elide
```

Установите проверенные новые бинарники и запустите:

```sh
sudo install -m 0755 elide elide-cli /usr/local/bin/
sudo systemctl start elide
elide-cli status
```

При обычном обновлении не удаляйте `/var/lib/elide`. Если кошелёк ноды
получает средства, отдельно сохраните `/var/lib/elide/wallet.key` и
защищайте его как приватный секрет.
