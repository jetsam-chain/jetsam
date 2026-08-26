# Локальный запуск Core

Архив Core предназначен для операторов нод, майнеров и разработчиков. В него входят:

- `elide` — полная нода со встроенным майнером;
- `jetsam-cli` — локальный клиент ноды и кошелька;
- `jetsam-miner` — внешний вычислитель PoW.

GUI-кошелёк распространяется отдельно.

## Скачивание и проверка

Выберите архив Core для своей платформы:

```text
jetsam-core-vVERSION-linux-x86_64.tar.gz
jetsam-core-vVERSION-linux-aarch64.tar.gz
jetsam-core-vVERSION-windows-x86_64.zip
jetsam-core-vVERSION-macos-aarch64.tar.gz
jetsam-core-vVERSION-macos-x86_64.tar.gz
```

Перед распаковкой сравните SHA-256 архива со значением в файле
`SHA256SUMS` из того же релиза.

## Проверка процессора

На Linux или macOS:

```sh
./elide --check-hardware
```

В Windows PowerShell:

```powershell
.\elide.exe --check-hardware
```

Последняя строка должна выглядеть так:

```text
NODE READY
```

Для рабочей ноды на x86-64 необходимы SSE4.1 и PCLMULQDQ, на ARM64 — NEON
и PMULL. При запуске автоматически выбирается `pclmul`, `avx2+vpclmul`,
`avx512bw+vpclmul` или `neon+pmull`.

Особенности виртуальных машин, требования к памяти и расчёт дискового
пространства описаны в разделе
[Аппаратные требования и ресурсы](../operate/hardware.md).

## Запуск обычной ноды

Запустите Core в текущем терминале:

```sh
./elide
```

При первом запуске будут созданы:

```text
~/.elide/elide.toml
~/.elide/data/
```

В другом терминале можно проверить состояние:

```sh
./jetsam-cli status
./jetsam-cli peers
./jetsam-cli state
```

По умолчанию P2P-нода слушает `0.0.0.0:9600`, а RPC доступен только на
`127.0.0.1:9601`.

Для корректной остановки выполните:

```sh
./jetsam-cli stop
```

Настройка постоянной системной службы, межсетевого экрана и обновлений
описана в разделе [Запуск ноды на Linux](../operate/node.md). Полный перечень
команд приведён в [справочнике CLI](../reference/cli.md).
