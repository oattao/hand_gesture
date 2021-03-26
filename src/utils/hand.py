def write_landmark_to_csv(landmark, datadf):
    row = []
    for point in landmark:
        row.append(point.x)
        row.append(point.y)
        row.append(point.z)
    datadf.loc[1] = row
    datadf.index = datadf.index + 1
    datadf = datadf.sort_index()
    return datadf

def parse_landmark_to_numpy_array(landmark):
    hand = []
    for point in landmark:
        hand.append(point.x)
        hand.append(point.y)
        hand.append(point.z)
    hand = np.array(hand)
    hand = np.expand_dims(hand, 0)
    return hand    