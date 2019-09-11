Creator "map2model Add-In v.1.8.1.0"
# ----- Input data -----
# Polygons layer: hams2_geol
# ROI layer: hamms_roi
# Faults layer: hams2_faults
# Unit field name: CODE
# Group field name: GROUP_
# Rocktype field name: ROCKTYPE1
# Fault buffer sze: 1E-06
# ----- Output data -----
# Number of groups: 10
# Number of units: 29
# Number of polygons: 184
# Number of faults: 102
# Unit contacts: 75
# Unit contacts length: 5470786.39099504
# Fault contacts: 0
# Fault contacts length: 0
# Igneous contacts: 4
# Igneous contacts length: 162324.399577232
graph [
  hierarchic 1
  directed 1

  node [
    id 0
    LabelGraphics [ text "Hamersley Group" anchor "n" fontStyle "bold" fontSize 14 ]
    isGroup 1
    graphics [ fill "#FAFAFA" ]
  ]

  node [
    id 2
    LabelGraphics [ text "Fortescue Group" anchor "n" fontStyle "bold" fontSize 14 ]
    isGroup 1
    graphics [ fill "#FAFAFA" ]
  ]

  node [
    id 6
    LabelGraphics [ text "A-mgn-PRK" anchor "n" fontStyle "bold" fontSize 14 ]
    isGroup 1
    graphics [ fill "#FAFAFA" ]
  ]

  node [
    id 13
    LabelGraphics [ text "Turee Creek Group" anchor "n" fontStyle "bold" fontSize 14 ]
    isGroup 1
    graphics [ fill "#FAFAFA" ]
  ]

  node [
    id 19
    LabelGraphics [ text "A-b-PRK" anchor "n" fontStyle "bold" fontSize 14 ]
    isGroup 1
    graphics [ fill "#FAFAFA" ]
  ]

  node [
    id 24
    LabelGraphics [ text "A-mgn-PMI" anchor "n" fontStyle "bold" fontSize 14 ]
    isGroup 1
    graphics [ fill "#FAFAFA" ]
  ]

  node [
    id 26
    LabelGraphics [ text "A-s-PMI" anchor "n" fontStyle "bold" fontSize 14 ]
    isGroup 1
    graphics [ fill "#FAFAFA" ]
  ]

  node [
    id 28
    LabelGraphics [ text "A-s-PRK" anchor "n" fontStyle "bold" fontSize 14 ]
    isGroup 1
    graphics [ fill "#FAFAFA" ]
  ]

  node [
    id 30
    LabelGraphics [ text "Shingle Creek Group" anchor "n" fontStyle "bold" fontSize 14 ]
    isGroup 1
    graphics [ fill "#FAFAFA" ]
  ]

  node [
    id 32
    LabelGraphics [ text "Wyloo Group" anchor "n" fontStyle "bold" fontSize 14 ]
    isGroup 1
    graphics [ fill "#FAFAFA" ]
  ]

  node [
    id 1
    LabelGraphics [ text "A-HAm-cib" fontSize 14 ]
    gid 0
    graphics [ fill "#FCD4B3" w 150 ]
  ]

  node [
    id 3
    LabelGraphics [ text "A-FOp-bs" fontSize 14 ]
    gid 2
    graphics [ fill "#DFC2FC" w 150 ]
  ]

  node [
    id 4
    LabelGraphics [ text "A-HAS-xsl-ci" fontSize 14 ]
    gid 0
    graphics [ fill "#FCB3B6" w 150 ]
  ]

  node [
    id 5
    LabelGraphics [ text "P_-HAb-cib" fontSize 14 ]
    gid 0
    graphics [ fill "#E9FCC7" w 150 ]
  ]

  node [
    id 7
    LabelGraphics [ text "A-mgn-PRK" fontSize 14 ]
    gid 6
    graphics [ fill "#FCB9B3" w 150 ]
  ]

  node [
    id 8
    LabelGraphics [ text "A-FOh-xs-f" fontSize 14 ]
    gid 2
    graphics [ fill "#C0EAFC" w 150 ]
  ]

  node [
    id 9
    LabelGraphics [ text "A-FOo-bbo" fontSize 14 ]
    gid 2
    graphics [ fill "#B8FCBF" w 150 ]
  ]

  node [
    id 10
    LabelGraphics [ text "A-FO-od" fontSize 14 ]
    gid 2
    graphics [ fill "#FCB3D1" w 150 ]
  ]

  node [
    id 11
    LabelGraphics [ text "A-HAd-kd" fontSize 14 ]
    gid 0
    graphics [ fill "#FCC6BB" w 150 ]
  ]

  node [
    id 12
    LabelGraphics [ text "A-FOj-xs-b" fontSize 14 ]
    gid 2
    graphics [ fill "#B8F0FC" w 150 ]
  ]

  node [
    id 14
    LabelGraphics [ text "P_-TKa-xs-k" fontSize 14 ]
    gid 13
    graphics [ fill "#FCB3EB" w 150 ]
  ]

  node [
    id 15
    LabelGraphics [ text "P_-HAo-ci" fontSize 14 ]
    gid 0
    graphics [ fill "#DED4FC" w 150 ]
  ]

  node [
    id 16
    LabelGraphics [ text "P_-HAj-xci-od" fontSize 14 ]
    gid 0
    graphics [ fill "#FAC2FC" w 150 ]
  ]

  node [
    id 17
    LabelGraphics [ text "P_-HAw-fr" fontSize 14 ]
    gid 0
    graphics [ fill "#E2FCBD" w 150 ]
  ]

  node [
    id 18
    LabelGraphics [ text "A-FO-xo-a" fontSize 14 ]
    gid 2
    graphics [ fill "#D2FCD6" w 150 ]
  ]

  node [
    id 20
    LabelGraphics [ text "A-b-PRK" fontSize 14 ]
    gid 19
    graphics [ fill "#FCCCCF" w 150 ]
  ]

  node [
    id 21
    LabelGraphics [ text "A-FOr-b" fontSize 14 ]
    gid 2
    graphics [ fill "#FCC0C8" w 150 ]
  ]

  node [
    id 22
    LabelGraphics [ text "A-FOu-bbo" fontSize 14 ]
    gid 2
    graphics [ fill "#C0C9FC" w 150 ]
  ]

  node [
    id 23
    LabelGraphics [ text "P_-TK-s" fontSize 14 ]
    gid 13
    graphics [ fill "#C8B3FC" w 150 ]
  ]

  node [
    id 25
    LabelGraphics [ text "A-mgn-PMI" fontSize 14 ]
    gid 24
    graphics [ fill "#FCEBB3" w 150 ]
  ]

  node [
    id 27
    LabelGraphics [ text "A-s-PMI" fontSize 14 ]
    gid 26
    graphics [ fill "#CBC2FC" w 150 ]
  ]

  node [
    id 29
    LabelGraphics [ text "A-s-PRK" fontSize 14 ]
    gid 28
    graphics [ fill "#C0FCDF" w 150 ]
  ]

  node [
    id 31
    LabelGraphics [ text "P_-SKq-stq" fontSize 14 ]
    gid 30
    graphics [ fill "#DAB8FC" w 150 ]
  ]

  node [
    id 33
    LabelGraphics [ text "P_-WYm-sp" fontSize 14 ]
    gid 32
    graphics [ fill "#B8E1FC" w 150 ]
  ]

  node [
    id 34
    LabelGraphics [ text "P_-WYa-st" fontSize 14 ]
    gid 32
    graphics [ fill "#CFE2FC" w 150 ]
  ]

  node [
    id 35
    LabelGraphics [ text "P_-WYd-kd" fontSize 14 ]
    gid 32
    graphics [ fill "#B6FCB3" w 150 ]
  ]

  node [
    id 36
    LabelGraphics [ text "P_-TKo-stq" fontSize 14 ]
    gid 13
    graphics [ fill "#D8FCCC" w 150 ]
  ]

  node [
    id 37
    LabelGraphics [ text "P_-TKk-sf" fontSize 14 ]
    gid 13
    graphics [ fill "#FCD9D7" w 150 ]
  ]

  node [
    id 38
    LabelGraphics [ text "P_-SKb-bb" fontSize 14 ]
    gid 30
    graphics [ fill "#CAD5FC" w 150 ]
  ]

  edge [
    source 5
    target 1
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 1
    target 12
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 15
    target 1
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 3
    target 9
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 18
    target 3
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 10
    target 3
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 22
    target 3
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 5
    target 4
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 4
    target 11
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 17
    target 5
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 5
    target 11
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 16
    target 5
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 5
    target 12
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 15
    target 5
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 5
    target 10
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 5
    target 22
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 8
    target 7
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 7
    target 20
    graphics [ style "line" arrow "last" width 3 fill "#FF0000" ]
  ]

  edge [
    source 21
    target 7
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 7
    target 29
    graphics [ style "line" arrow "last" width 3 fill "#FF0000" ]
  ]

  edge [
    source 9
    target 8
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 8
    target 20
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 8
    target 21
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 18
    target 8
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 8
    target 29
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 10
    target 12
    graphics [ style "line" arrow "both" width 3 fill "#0000FF" ]
  ]

  edge [
    source 12
    target 22
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 4
    target 12
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 14
    target 31
    graphics [ style "line" arrow "both" width 3 fill "#0000FF" ]
  ]

  edge [
    source 14
    target 36
    graphics [ style "line" arrow "both" width 3 fill "#0000FF" ]
  ]

  edge [
    source 14
    target 37
    graphics [ style "line" arrow "both" width 3 fill "#0000FF" ]
  ]

  edge [
    source 11
    target 12
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 15
    target 17
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 15
    target 23
    graphics [ style "line" arrow "both" width 3 fill "#0000FF" ]
  ]

  edge [
    source 17
    target 16
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 23
    target 17
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 11
    target 1
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 18
    target 10
    graphics [ style "line" arrow "both" width 3 fill "#FF0000" ]
  ]

  edge [
    source 18
    target 22
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 10
    target 8
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 8
    target 25
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 8
    target 27
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 10
    target 22
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 4
    target 1
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 11
    target 10
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 1
    target 10
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 18
    target 9
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 15
    target 16
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 23
    target 16
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 15
    target 12
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 23
    target 12
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 21
    target 20
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 15
    target 37
    graphics [ style "line" arrow "both" width 3 fill "#0000FF" ]
  ]

  edge [
    source 33
    target 15
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 10
    target 9
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 25
    target 27
    graphics [ style "line" arrow "last" width 3 fill "#FF0000" ]
  ]

  edge [
    source 37
    target 17
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 22
    target 9
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 31
    target 37
    graphics [ style "line" arrow "both" width 3 fill "#0000FF" ]
  ]

  edge [
    source 31
    target 38
    graphics [ style "line" arrow "both" width 3 fill "#0000FF" ]
  ]

  edge [
    source 33
    target 31
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 23
    target 31
    graphics [ style "line" arrow "both" width 3 fill "#0000FF" ]
  ]

  edge [
    source 35
    target 23
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 33
    target 23
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 23
    target 22
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 35
    target 31
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 31
    target 22
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 33
    target 35
    graphics [ style "line" arrow "both" width 3 fill "#0000FF" ]
  ]

  edge [
    source 33
    target 22
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 34
    target 35
    graphics [ style "line" arrow "both" width 3 fill "#0000FF" ]
  ]

  edge [
    source 35
    target 22
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 36
    target 37
    graphics [ style "line" arrow "both" width 3 fill "#0000FF" ]
  ]

  edge [
    source 37
    target 38
    graphics [ style "line" arrow "both" width 3 fill "#0000FF" ]
  ]

  edge [
    source 33
    target 37
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

  edge [
    source 33
    target 38
    graphics [ style "line" arrow "last" width 3 fill "#0000FF" ]
  ]

]
