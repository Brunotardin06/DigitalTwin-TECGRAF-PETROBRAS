import os, zipfile, hashlib, unicodedata
from pathlib import Path

def listarZips(path):
    return [nome for nome in os.listdir(path) if nome.lower().endswith(".zip")]


def listarPastas(path):
    return [nome for nome in os.listdir(path) if (Path(path) / nome).is_dir()]


def paraCaminhoLongoWindows(path):
    pathStr = str(Path(path).resolve())
    if os.name != "nt":
        return pathStr
    if pathStr.startswith("\\\\?\\"):
        return pathStr
    if pathStr.startswith("\\\\"):
        return "\\\\?\\UNC\\" + pathStr[2:]
    return "\\\\?\\" + pathStr


def nomeDestinoCurto(caminhoZip):
    stem = Path(caminhoZip).stem
    prefixo = stem[:40].rstrip()
    sufixo = hashlib.md5(stem.encode("utf-8")).hexdigest()[:8]
    return f"{prefixo}_{sufixo}"


def chaveFlexivel(texto):
    textoNorm = unicodedata.normalize("NFKD", texto)
    semAcentos = "".join(ch for ch in textoNorm if not unicodedata.combining(ch))
    return "".join(ch for ch in semAcentos.lower() if ch.isalnum())


def resolverCaminhoEntrada(path):
    p = Path(path)
    if p.exists():
        return p

    parent = p.parent
    alvo = p.name
    if not parent.exists() or not parent.is_dir():
        return p

    chaveAlvo = chaveFlexivel(alvo)
    for candidato in parent.iterdir():
        if chaveFlexivel(candidato.name) == chaveAlvo:
            return candidato

    return p


def extrairZipNoMesmoNivel(path, nomeZip):
    caminhoZip = Path(path) / nomeZip
    pastaDestino = caminhoZip.parent / nomeDestinoCurto(caminhoZip)
    pastaDestino.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(paraCaminhoLongoWindows(caminhoZip), "r") as zipRef:
        zipRef.extractall(paraCaminhoLongoWindows(pastaDestino))

    print(f"Extraído: {nomeZip}")
    print(f"Destino: {pastaDestino}\n")


def menuExtracao(path):
    path = resolverCaminhoEntrada(path)
    if not os.path.isdir(path):
        print(f"Caminho inválido: {path}")
        return

    atual = Path(path)
    raiz = Path(path)

    while True:
        pastas = listarPastas(atual)
        arquivos = listarZips(atual)

        print("\n=== MENU DE NAVEGAÇÃO E EXTRAÇÃO ===")
        print(f"Nível atual: {atual}")

        if pastas:
            print("\nPastas:")
            for i, nome in enumerate(pastas, start=1):
                print(f"P{i}: {nome}")
        else:
            print("\nPastas: nenhuma")

        if arquivos:
            print("\nZIPs:")
            for i, nome in enumerate(arquivos, start=1):
                print(f"Z{i}: {nome}")
        else:
            print("\nZIPs: nenhum")

        print("\n1 - Entrar em pasta (ID P)")
        print("2 - Voltar um nível")
        print("3 - Extrair todos os .zip deste nível")
        print("4 - Extrair um .zip por ID (Z)")
        print("0 - Sair")
        opcao = input("Escolha uma opção: ").strip()

        if opcao == "0":
            print("Encerrado.")
            break

        if opcao == "1":
            if not pastas:
                print("Nenhuma pasta para entrar neste nível.")
                continue

            try:
                indexPasta = int(input("Digite o ID da pasta (número após P): "))
            except ValueError:
                print("ID inválido.")
                continue

            if indexPasta < 1 or indexPasta > len(pastas):
                print("ID fora do intervalo.")
                continue

            atual = atual / pastas[indexPasta - 1]
            continue

        if opcao == "2":
            if atual == raiz:
                print("Você já está no diretório raiz.")
            else:
                atual = atual.parent
            continue

        if opcao == "3":
            if not arquivos:
                print("Nenhum arquivo .zip encontrado neste nível.")
                continue

            for nomeZip in arquivos:
                extrairZipNoMesmoNivel(atual, nomeZip)
            print("Extração concluída.")

        elif opcao == "4":
            if not arquivos:
                print("Nenhum arquivo .zip encontrado neste nível.")
                continue

            try:
                indexExtract = int(input("Digite o ID do zip (número após Z): "))
            except ValueError:
                print("ID inválido.")
                continue

            if indexExtract < 1 or indexExtract > len(arquivos):
                print("ID fora do intervalo.")
                continue

            extrairZipNoMesmoNivel(atual, arquivos[indexExtract - 1])

        else:
            print("Opção inválida.")


if __name__ == "__main__":
    caminho = input("Informe o diretório raiz: ").strip()
    menuExtracao(caminho)