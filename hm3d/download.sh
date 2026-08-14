#!/bin/bash

# 다운로드할 파일 목록과 URL 정의
# (질문에서 제공된 테이블 내용을 기반으로 리스트를 구성했습니다.)
declare -A HM3D_FILES

HM3D_FILES=(
    ["hm3d-minival-glb-v0.2.tar"]="https://api.matterport.com/resources/habitat/hm3d-minival-glb-v0.2.tar"
    ["hm3d-minival-habitat-v0.2.tar"]="https://api.matterport.com/resources/habitat/hm3d-minival-habitat-v0.2.tar"
    ["hm3d-minival-semantic-annots-v0.2.tar"]="https://api.matterport.com/resources/habitat/hm3d-minival-semantic-annots-v0.2.tar"
    ["hm3d-minival-semantic-configs-v0.2.tar"]="https://api.matterport.com/resources/habitat/hm3d-minival-semantic-configs-v0.2.tar"
    ["hm3d-train-glb-v0.2.tar"]="https://api.matterport.com/resources/habitat/hm3d-train-glb-v0.2.tar"
    ["hm3d-train-habitat-v0.2.tar"]="https://api.matterport.com/resources/habitat/hm3d-train-habitat-v0.2.tar"
    ["hm3d-train-semantic-annots-v0.2.tar"]="https://api.matterport.com/resources/habitat/hm3d-train-semantic-annots-v0.2.tar"
    ["hm3d-train-semantic-configs-v0.2.tar"]="https://api.matterport.com/resources/habitat/hm3d-train-semantic-configs-v0.2.tar"
    ["hm3d-val-glb-v0.2.tar"]="https://api.matterport.com/resources/habitat/hm3d-val-glb-v0.2.tar"
    ["hm3d-val-habitat-v0.2.tar"]="https://api.matterport.com/resources/habitat/hm3d-val-habitat-v0.2.tar"
    ["hm3d-val-semantic-annots-v0.2.tar"]="https://api.matterport.com/resources/habitat/hm3d-val-semantic-annots-v0.2.tar"
    ["hm3d-val-semantic-configs-v0.2.tar"]="https://api.matterport.com/resources/habitat/hm3d-val-semantic-configs-v0.2.tar"
    ["hm3d-example-glb-v0.2.tar"]="https://raw.githubusercontent.com/matterport/habitat-matterport-3dresearch/master/example/hm3d-example-glb-v0.2.tar"
    ["hm3d-example-habitat-v0.2.tar"]="https://raw.githubusercontent.com/matterport/habitat-matterport-3dresearch/master/example/hm3d-example-habitat-v0.2.tar"
    ["hm3d-example-semantic-annots-v0.2.tar"]="https://raw.githubusercontent.com/matterport/habitat-matterport-3dresearch/master/example/hm3d-example-semantic-annots-v0.2.tar"
    ["hm3d-example-semantic-configs-v0.2.tar"]="https://raw.githubusercontent.com/matterport/habitat-matterport-3dresearch/master/example/hm3d-example-semantic-configs-v0.2.tar"
)

# 데이터셋을 저장할 디렉토리 생성
OUTPUT_DIR="./hm3d_dataset"
mkdir -p "$OUTPUT_DIR"

HTTP_COOKIE="__cf_bm=AictrUVNS.95hibyPXHfWz3x6GqqNrWoS5j30pNfyxg-1783581218.5498438-1.0.1.1-w7ZvjDBYkadWlHOnkkZydZigMOfJevjLOmANPTjr8.yB.027HwmhW48yU0PORdMsi3TgsyvCRjG47Y7C4G0nGV6levi7QUGeRuk4gwhbAqIpXNcwxTMnEYZY5bZbgLnq; MPBillingSession=XO6Lxv3yZgbRMsFwy6RDa; ajs_anonymous_id=f41d2e41-7654-4179-a3d2-2e2b5205f9f5; domain_token=f47b0c80eb0d4f42bb02ded34f4a8b08; authn_token=73dbc7ce404f43a0a5c2881d4c2e0e3e; cookie_consent_v3={%22version%22:3%2C%22strictlyNecessary%22:true%2C%22custom%22:{%22performance%22:true%2C%22functionality%22:true%2C%22targeting%22:true}%2C%22usChecked%22:true}; mp_membership_sid=bKFbUBoXZEZ; intercom-device-id-toxdrc11=c04e8058-eb6c-488a-8f67-8109341b4023; osano_consentmanager_uuid=38575de0-6f77-4864-bb8d-94f066a209ec; osano_consentmanager=oz7YTm2AsPLjgyaT5OWXuhNrhkACBGE1v73tNhsBvwGMULvT-Gyp6lul5lmzBUnO39eYQkXLhsgzcb_lnBE-bSbZxiielPtcwQHfLs5gcB2InPsDUAKZswlDuKfZszOs-U1Tk1uE98P-2Zcu7cv0IgE303Q20QldGI-uwjlmQtn6m21tYLXIVRXzMYOhoAqiDdS-RItOcDwQ-V6lodnBJEFrHXkrmuF958peHQW9LAWs-RolX3ictrswT67_iOHavKXMg_dY5ZDF4ZT9hggQJQcd4oJJIBzWwfTe_QBO7EAa76fgZXE3Z0z0xbIP0cp-; intercom-session-toxdrc11=R3M3UUowR3U2MUJtVGR5VU1wWktncDFYMmhaRUJzRkRZMXM4M3ovU2J1SEd0SUNzbG43N3pkdGYzdEtLdU5mU0daL1FkZmVCYUpOSWxrMUlQVWNTcG51VmJ4S2x6MTAxMi9TQVFBb2hUemx4bXlSMHR6c2RPeHE2TnRPcVVuQlk4VUZ1QUxFQTBLMUkrbnJIME5tQzZyY1NQV24vSER4R1BHcTB0c2FrZkZ3NFNvWUp2YnJxNVQrMng2Qm96VHNnSUs4U2pEZTlpaTQrT2FsZ01xK3ZPemtmcEpabzN6S3ZUeEU3TkkwRTQ0bExsMTlmTWE1b2ZaTEllSEZIdVNrc1NQL3V0Wm1SZmdEeEl4UzBON254NlE9PS0tblJlODlDNUNqcUdqM1RxTnl5VGVEUT09--3291985838a7258881321278a5ade45efdde6df0"

echo "Starting HM3D dataset download..."

# 루프를 돌며 wget으로 순차적 다운로드 진행
for FILE_NAME in "${!HM3D_FILES[@]}"; do
    URL="${HM3D_FILES[$FILE_NAME]}"
    TARGET_PATH="$OUTPUT_DIR/$FILE_NAME"
    
    # 이미 파일이 존재하는 경우 다운로드를 건너뜀
    if [ -f "$TARGET_PATH" ]; then
        echo "[Skip] $FILE_NAME already exists."
    else
        echo "[Download] Fetching $FILE_NAME..."
        # 이어받기 기능(-c)을 활성화하여 wget 실행
        wget -c "$URL" -O "$TARGET_PATH" --header="Cookie: $HTTP_COOKIE"
    fi
done

echo "All download processes finished."
