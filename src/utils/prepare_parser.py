"""This script contains function to add feature arguments to the parser
"""


def add_feature_arguments(parser):
    # Features computations require a split id
    parser.add_argument('--split_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, help="Dataset to prepare.")

    # One-hot features:
    
    parser.add_argument("-streak", action="store_true", help="Include streak feature")


    parser.add_argument('-u', action='store_true',
                        help='If True, user one hot encoding.')
    parser.add_argument('-i', action='store_true',
                        help='If True, item one hot encoding.')
    parser.add_argument('-s', action='store_true',
                        help='If True, skills many hot encoding .')
    parser.add_argument('-sch', action='store_true',
                        help='If True, school one hot encoding.')
    parser.add_argument('-tea', action='store_true',
                        help='If True, teacher one hot encoding.')
    parser.add_argument('-sm', action='store_true',
                        help='If True, s_modules one hot encoding.')
    parser.add_argument('-c', action='store_true',
                        help='If True, course_id one hot encoding.')
    parser.add_argument('-d', action='store_true',
                        help='If True, item difficulty one hot encoding.')
    parser.add_argument('-at', action='store_true',
                        help='If True, item app type one hot encoding.')
    parser.add_argument('-t', action='store_true',
                        help='If True, item topic one hot encoding.')
    parser.add_argument('-bundle', action='store_true',
                        help='If True, item bundle one hot encoding.')
    parser.add_argument('-part', action='store_true',
                        help='If True, item part one hot encoding.')
    parser.add_argument('-ss', action='store_true',
                        help='If True, social support one hot encoding.')
    parser.add_argument('-age', action='store_true',
                        help='If True, user age one hot encoding.')
    parser.add_argument('-gender', action='store_true',
                        help='If True, user gender one hot encoding.')
    parser.add_argument('-user_skill', action='store_true',
                        help='If True, user skill one hot encoding.')

    # User history features:
    parser.add_argument('-tcA', action='store_true',
                        help='If True, total count of attempts.')
    parser.add_argument('-tcW', action='store_true',
                        help='If True, total count of wins.')
    parser.add_argument('-scA', action='store_true',
                        help='If True, skill count of attempts.')
    parser.add_argument('-scW', action='store_true',
                        help='If True, skill count of wins.')
    parser.add_argument('-icA', action='store_true',
                        help='If True, item count of attempts.')
    parser.add_argument('-icW', action='store_true',
                        help='If True, item count of wins.')
    parser.add_argument('-partcA', action='store_true',
                        help='If True, TOIEC part count of attempts.')
    parser.add_argument('-partcW', action='store_true',
                        help='If True, TOIEC part count of wins.')

    # Graph features
    parser.add_argument('-pre', action='store_true',
                        help='If True, prereq skill one-hot encoding.')
    parser.add_argument('-post', action='store_true',
                        help='If True, postreq skill one-hot encoding.')
    parser.add_argument('-precA', action='store_true',
                        help='If True, pre-req skill count attempts.')
    parser.add_argument('-precW', action='store_true',
                        help='If True, skill pre-req of wins.')
    parser.add_argument('-postcA', action='store_true',
                        help='If True, post-req skill count attempts.')
    parser.add_argument('-postcW', action='store_true',
                        help='If True, post-req skill count of wins.')

    # Difficulty features:
    parser.add_argument('-acSd', action='store_true',
                        help='If True, avg correct skill level difficulty.')
    parser.add_argument('-acTd', action='store_true',
                        help='If True, avg correct topic level difficulty.')
    parser.add_argument('-acCd', action='store_true',
                        help='If True, avg correct course level difficulty.')
    parser.add_argument('-aiSd', action='store_true',
                        help='If True, avg incorrect skill level difficulty.')
    parser.add_argument('-aiTd', action='store_true',
                        help='If True, avg incorrect topic level difficulty.')
    parser.add_argument('-aiCd', action='store_true',
                        help='If True, avg incorrect course level difficulty.')
    parser.add_argument('-acSdr', action='store_true',
                        help='If True, avg correct skill level difficulty \
                            ratio with the current difficulty.')
    parser.add_argument('-acTdr', action='store_true',
                        help='If True, avg correct topic level difficulty \
                            ratio with the current difficulty.')
    parser.add_argument('-acCdr', action='store_true',
                        help='If True, avg correct course level difficulty \
                            ratio with the current difficulty.')
    parser.add_argument('-aiSdr', action='store_true',
                        help='If True, avg incorrect skill level difficulty \
                            ratio with the current difficulty.')
    parser.add_argument('-aiTdr', action='store_true',
                        help='If True, avg incorrect topic level difficulty \
                            ratio with the current difficulty.')
    parser.add_argument('-aiCdr', action='store_true',
                        help='If True, avg incorrect course level difficulty \
                            ratio with the current difficulty.')
    parser.add_argument('-db', action='store_true',
                        help='If True, separate difficulties into 3 buckets \
                            [10-20], [30-70], [80-90].')

    # Video features:
    parser.add_argument('-vw', action='store_true',
                        help='If True, count user watched videos on skills.')
    parser.add_argument('-vs', action='store_true',
                        help='If True, count user skipped videos on skills.')
    parser.add_argument('-vt', action='store_true',
                        help='If True, user time spent watching videos.')

    # Reading features:
    parser.add_argument('-rc', action='store_true',
                        help='If True, count read explanations skill & total.')
    parser.add_argument('-rt', action='store_true',
                        help='If True, reading time skill & total.')

    # Study module features:
    parser.add_argument('-smA', action='store_true',
                        help='If True, count user attempts of study module.')
    parser.add_argument('-smW', action='store_true',
                        help='If True, count user wins of study module.')

    # Interaction time features
    parser.add_argument('-resp_time', action='store_true',
                        help='If True, user response time in seconds.')
    parser.add_argument('-resp_time_cat', action='store_true',
                        help='If True, user response time phi and categories.')
    parser.add_argument('-prev_resp_time_cat', action='store_true',
                        help='If True, user prior response time phi and cat.')

    parser.add_argument('-lag_time', action='store_true',
                        help='If True, user lag time in second.')
    parser.add_argument('-lag_time_cat', action='store_true',
                        help='If True, user lag time phi and categories.')
    parser.add_argument('-prev_lag_time_cat', action='store_true',
                        help='If True, user prior lag time phi and cat.')

    # Time Window features
    parser.add_argument('-tcA_TW', action='store_true',
                        help='If True, total count attemts in time windows.')
    parser.add_argument('-tcW_TW', action='store_true',
                        help='If True, total count wins in time windows.')
    parser.add_argument('-scA_TW', action='store_true',
                        help='If True, skill count attemts in time windows.')
    parser.add_argument('-scW_TW', action='store_true',
                        help='If True, skill count wins in time windows.')
    parser.add_argument('-icA_TW', action='store_true',
                        help='If True, item count attemts in time windows.')
    parser.add_argument('-icW_TW', action='store_true',
                        help='If True, item count wins in time windows.')

    # RPFA features
    parser.add_argument('-rpfa_F', action='store_true',
                        help='If True, recency-weighted failure count.')
    parser.add_argument('-rpfa_R', action='store_true',
                        help='If True, recency-weighted proportion correct.')

    # PPE features
    parser.add_argument('-ppe', action='store_true',
                        help='If True, space-weighted attempt count.')

    # Datetime features
    parser.add_argument('-month', action='store_true',
                        help='If True, month one-hot.')
    parser.add_argument('-week', action='store_true',
                        help='If True, week one-hot.')
    parser.add_argument('-day', action='store_true',
                        help='If True, day one-hot.')
    parser.add_argument('-hour', action='store_true',
                        help='If True, hour one-hot.')
    parser.add_argument('-weekend', action='store_true',
                        help='If True, weekend one-hot.')
    parser.add_argument('-part_of_day', action='store_true',
                        help='If True, part of day one-hot.')

    # Higher-order features:
    parser.add_argument('-user_avg_correct', action='store_true',
                        help='If True, user average correctness over time.')
    parser.add_argument('-n_gram', action='store_true',
                        help='If True, correctness patterns in sequence.')

    # Global features:
    parser.add_argument('-gi', action='store_true',
                        help='If True, item average correctness over time.')
    parser.add_argument('-gs', action='store_true',
                        help='If True, skill average correctness over time.')
    parser.add_argument('-gsch', action='store_true',
                        help='If True, school average correctness over time.')

    # Other:
    parser.add_argument('-ones', action='store_true',
                        help='If True, adds a one vector.')
